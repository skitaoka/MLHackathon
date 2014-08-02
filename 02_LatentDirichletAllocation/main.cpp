#include <iostream>
#include <fstream>

#include <memory>
#include <numeric>
#include <algorithm>
#include <random>
#include <functional>

#include <string>
#include <map>
#include <vector>

namespace reactor
{
  namespace statistics
  {
    // 累積密度関数 [first, last) からサンプリング
    template <typename Iterator>
    std::size_t sample(Iterator first, Iterator last, std::mt19937_64 & engine)
    {
      if (first == last) {
        return 0;
      }
      std::size_t const size = std::distance(first, last);

      typedef std::iterator_traits<Iterator>::value_type value_type;
      std::uniform_real_distribution<value_type> runif(value_type(), first[size - 1]);
      return std::distance(first, std::lower_bound(first, last, runif(engine)) - 1);
    }
  }
}

// cf. http://d.hatena.ne.jp/n_shuyo/20110215/lda
// w: 単語
// n: w の n[m] から n[m+1]-1 までが m-th 文章の単語
// M: 文章数
// V: 語彙数
// K: トピック数
// a: ハイパーパラメータ
// b: ハイパーパラメータ
std::pair<std::vector<double>, std::vector<double>>
  lda(
    std::vector<std::size_t> const & w,
    std::vector<std::size_t> const & n,
    std::size_t const M,
    std::size_t const V,
    std::size_t const K,
    double      const alpha,
    double      const beta,
    std::size_t const sample_size)
{
  std::mt19937_64 engine(std::random_device("")());

  // 各単語のトピック
  std::vector<std::size_t> z(w.size());
  {
    std::uniform_int_distribution<std::size_t> runif(0, K-1);
    for (std::size_t i = 0, size = z.size(); i < size; ++i) {
      z[i] = runif(engine);
    }
  }

  // n_mk[m*K+k]: m-th 文章の k-th トピックの単語数
  std::vector<std::size_t> n_mk(K*M);
  for (std::size_t m = 0; m < M; ++m) {
    for (std::size_t i = n[m], count = n[m + 1]; i < count; ++i) {
      std::size_t const k = z[i];
      ++n_mk[m * K + k];
    }
  }

  // n_vk[v*K+k]: k-th トピックの v-th 単語数
  std::vector<std::size_t> n_vk(K*V);
  for (std::size_t i = 0, size = w.size(); i < size; ++i) {
    std::size_t const v = w[i];
    std::size_t const k = z[i];
    ++n_vk[v * K + k];
  }

  // n_k[k]: k-th トピックの単語数
  std::vector<std::size_t> n_k(K);
  for (std::size_t i = 0, size = z.size(); i < size; ++i) {
    std::size_t const k = z[i];
    ++n_k[k];
  }

  // ギブスサンプリング
  for (std::size_t s = 0; s < sample_size; ++s) {
    for (std::size_t m = 0; m < M; ++m) {
      for (std::size_t i = n[m], count = n[m + 1]; i < count; ++i) {
        std::size_t const v = w[i];
        std::size_t const k = z[i]; // いまのトピック

        --n_mk[m * K + k];
        --n_vk[v * K + k];
        --n_k[k];

        // トピック分布の累積分布関数をつくる
        std::vector<double> theta(K + 1);
        for (std::size_t t = 0; t < K; ++t) {
          theta[t + 1] = theta[t]
            + (n_mk[m * K + t] + alpha)
            * (n_vk[v * K + k] + beta)
            / (n_k[t] + V * beta);
        }

        // あたらしいトピックをサンプリング
        std::size_t const t = reactor::statistics::sample(
          std::begin(theta), std::end(theta), engine);

        ++n_mk[m * K + t];
        ++n_vk[v * K + t];
        ++n_k[t];

        z[i] = t; // トピックを新しく設定
      }
    }
  }

  // トピックごとの単語分布をもとめる
  std::vector<double> phi_kv(K * V);
  for (std::size_t k = 0; k < K; ++k) {
    for (std::size_t v = 0; v < V; ++v) {
      phi_kv[k * V + v] = (n_vk[v * K + k] + beta) / (n_k[k] + V * beta);
    }
  }

  // 文章ごとのトピック分布をもとめる
  std::vector<double> theta_mk(M * K);
  for (std::size_t m = 0; m < M; ++m) {
    std::size_t const n_m = n[m + 1] - n[m];
    for (std::size_t k = 0; k < K; ++k) {
      theta_mk[m * K + k] = (n_mk[m * K + k] + alpha) / (n_m + K * alpha);
    }
  }

  return std::make_pair(std::move(theta_mk), std::move(phi_kv));
}

int main(int argc, char* argv[])
try
{
  // 人工的にデータをつくる
  std::size_t const M =  100; // 文章数
  std::size_t const V = 1000; // 語彙数
  std::size_t const K =   10; // トピック数

  double const alpha = 0.5;
  double const beta = 2.0;

  std::mt19937_64 engine(std::random_device("")());

  std::vector<std::size_t> w(M*V); // 単語
  std::vector<std::size_t> n(M + 1);
  for (std::size_t m = 1; m <= M; ++m) {
    n[m] = w.size() * m / M;
  }

  // トピックごとの累積単語分布をつくる
  std::vector<double> phi(K * (V + 1));
  {
    std::gamma_distribution<double> rgamma(beta, 1.0);
    for (std::size_t k = 0; k < K; ++k) {
      for (std::size_t v = 0; v < V; ++v) {
        phi[k * (V + 1) + (v + 1)] = phi[k * (V + 1) + v] + rgamma(engine);
      }
    }
  }

  // 文章ごとの累積トピック分布をつくる
  std::vector<double> theta(M * (K + 1));
  {
    std::gamma_distribution<double> rgamma(alpha, 1.0);
    for (std::size_t m = 0; m < M; ++m) {
      for (std::size_t k = 0; k < K; ++k) {
        theta[m * (K + 1) + (k + 1)] = theta[m * (K + 1) + k] + rgamma(engine);
      }
    }
  }

  for (std::size_t m = 0; m < M; ++m) {
    for (std::size_t i = n[m], count = n[m + 1]; i < count; ++i) {
      // トピックをサンプリング
      std::size_t const k = reactor::statistics::sample(
        std::begin(theta) + ((m + 0) * (K + 1)),
        std::begin(theta) + ((m + 1) * (K + 1)), engine);

      // k-th トピックの単語分布から単語 v をサンプリング
      std::size_t const v = reactor::statistics::sample(
        std::begin(phi) + ((k + 0) * (V + 1)),
        std::begin(phi) + ((k + 1) * (V + 1)), engine);

      w[i] = v;
    }
  }

  auto const theta_phi = lda(w, n, M, V, K, alpha, beta, 10000);

  // 文章ごとのトピック分布
  for (std::size_t m = 0; m < M; ++m) {
    std::cout << m << "-th document:";
    for (std::size_t k = 0; k < K; ++k) {
      std::cout << ' ' << theta_phi.first[m * K + k];
    }
    std::cout << '\n';
  }

  // トピックごとの単語分布をもとめる
  for (std::size_t k = 0; k < K; ++k) {
    std::cout << k << "-th topic:";
    for (std::size_t v = 0; v < V; ++v) {
      std::cout << ' ' << theta_phi.second[k * V + v];
    }
    std::cout << '\n';
  }

  return 0;
}
catch (std::exception const & e)
{
  std::cerr << e.what() << '\n';

  return 1;
}