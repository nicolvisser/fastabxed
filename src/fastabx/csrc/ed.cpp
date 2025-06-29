#include <ATen/Parallel.h>
#include <Python.h>
#include <torch/library.h>
#include <torch/types.h>

extern "C" {
/* Creates a dummy empty _C module that can be imported from Python.
   The import from Python will load the .so consisting of this file
   in this extension, so that the TORCH_LIBRARY static initializers
   below are run. */
PyObject* PyInit__C(void) {
  static struct PyModuleDef module_def = {
      PyModuleDef_HEAD_INIT,
      "_C", /* name of module */
      NULL, /* module documentation, may be NULL */
      -1, /* size of per-interpreter state of the module,
             or -1 if the module keeps state in global variables. */
      NULL, /* methods */
  };
  return PyModule_Create(&module_def);
}
}

namespace fastabx {

std::vector<int64_t> remove_consecutive_duplicates(const torch::TensorAccessor<int64_t, 1>& acc, int64_t size) {
  std::vector<int64_t> result;
  if (size == 0) return result;
  result.push_back(acc[0]);
  for (int64_t i = 1; i < size; ++i) {
    if (acc[i] != acc[i - 1]) {
      result.push_back(acc[i]);
    }
  }
  return result;
}

float _ed_cpu(torch::Tensor a, torch::Tensor b) {
  const auto N = a.size(0);
  const auto M = b.size(0);
  TORCH_CHECK(N > 0 && M > 0, "Empty input tensor");

  auto a_acc = a.accessor<int64_t, 1>();
  auto b_acc = b.accessor<int64_t, 1>();

  // Remove consecutive duplicates
  std::vector<int64_t> a_vec = remove_consecutive_duplicates(a_acc, N);
  std::vector<int64_t> b_vec = remove_consecutive_duplicates(b_acc, M);
  int64_t N2 = a_vec.size();
  int64_t M2 = b_vec.size();

  // Create DP table
  std::vector<std::vector<int64_t>> dp(N2 + 1, std::vector<int64_t>(M2 + 1, 0));
  for (int64_t i = 0; i <= N2; ++i) dp[i][0] = i;
  for (int64_t j = 0; j <= M2; ++j) dp[0][j] = j;

  for (int64_t i = 1; i <= N2; ++i) {
    for (int64_t j = 1; j <= M2; ++j) {
      int64_t cost = (a_vec[i - 1] == b_vec[j - 1]) ? 0 : 1;
      dp[i][j] = std::min({dp[i - 1][j] + 1,        // deletion
                           dp[i][j - 1] + 1,        // insertion
                           dp[i - 1][j - 1] + cost  // substitution
                          });
    }
  }
  int64_t ed = dp[N2][M2];
  int64_t norm = std::max(N2, M2);
  return norm > 0 ? static_cast<float>(ed) / norm : 0.0f;
}

torch::Tensor ed_cpu(torch::Tensor a, torch::Tensor b) {
  const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(a.device());
  return torch::tensor(_ed_cpu(a, b), options);
}

torch::Tensor ed_batch_cpu(torch::Tensor x, torch::Tensor y, torch::Tensor sx, torch::Tensor sy, bool symmetric) {
  const auto nx = x.size(0);
  const auto ny = y.size(0);
  const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
  const auto sx_a = sx.accessor<int64_t, 1>();
  const auto sy_a = sy.accessor<int64_t, 1>();
  auto out = torch::zeros({nx, ny}, options);
  auto out_a = out.accessor<float, 2>();

  at::parallel_for(0, nx, 1, [&](int64_t start, int64_t end) {
    for (int64_t i = start; i < end; ++i) {
      int64_t start_j = symmetric ? i : 0;
      for (int64_t j = start_j; j < ny; ++j) {
        if (symmetric && i == j)
          continue;
        auto a = x[i].slice(0, 0, sx_a[i]);
        auto b = y[j].slice(0, 0, sy_a[j]);
        out_a[i][j] = _ed_cpu(a, b);
        if (symmetric && i != j) {
          out_a[j][i] = out_a[i][j];
        }
      }
    }
  });
  return out;
}

TORCH_LIBRARY(fastabx, m) {
  m.def("ed(Tensor x, Tensor y) -> Tensor", {torch::Tag::pt2_compliant_tag});
  m.def("ed_batch(Tensor x, Tensor y, Tensor sx, Tensor sy, bool symmetric) -> Tensor", {torch::Tag::pt2_compliant_tag});
}

TORCH_LIBRARY_IMPL(fastabx, CPU, m) {
  m.impl("ed", &ed_cpu);
  m.impl("ed_batch", &ed_batch_cpu);
}

} // namespace fastabx