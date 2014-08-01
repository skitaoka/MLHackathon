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

// cf. http://d.hatena.ne.jp/n_shuyo/20110215/lda
// w: 単語
// n: w の n[m] から n[m+1]-1 までが m-th 文章の単語
// M: 文章数
// V: 語彙数
// K: トピック数
// a: ハイパーパラメータ
// b: ハイパーパラメータ
void lda(
  std::vector<std::size_t> const & w,
  std::vector<std::size_t> const & n,
  std::size_t const M,
  std::size_t const V,
  std::size_t const K,
  double      const alpha,
  double      const beta,
  std::size_t const burn_in,
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

  // 単語ごとのトピックの頻度
  std::vector<std::size_t> q(w.size() * K);
  for (std::size_t s = 0, ss = burn_in + sample_size; s < ss; ++s) {
    // ギブスサンプリング
    for (std::size_t m = 0; m < M; ++m) {
      for (std::size_t i = n[m], count = n[m + 1]; i < count; ++i) {
        std::size_t const v = w[i];
        std::size_t const k = z[i]; // いまのトピック

        --n_mk[m * K + k];
        --n_vk[v * K + k];
        --n_k[k];

        // 確率質量関数 (トピック分布) をつくる
        std::vector<double> theta(K + 1);
        for (std::size_t t = 0; t < K; ++t) {
          theta[t + 1]
            = (n_mk[m * K + t] + alpha)
            * (n_vk[v * K + k] + beta)
            / (n_k[t] + V * beta);
        }

        // 累積分布関数をつくる
        for (std::size_t t = 0; t < K; ++t) {
          theta[t + 1] += theta[t];
        }

        // あたらしいトピックをサンプリング
        std::size_t t = 0;
        for (double const xi = std::uniform_real_distribution<double>(0.0, theta[K])(engine); xi > theta[t + 1]; ++t);

        ++n_mk[m * K + t];
        ++n_vk[v * K + t];
        ++n_k[t];
      }
    }

    if (s >= burn_in) {
      // カウント
      for (std::size_t i = 0, size = z.size(); i < size; ++i) {
        std::size_t const k = z[i];
        ++q[i * K + k];
      }
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

  // トピックごとの単語分布をもとめる
  std::vector<double> phi_kv(K * V);
  for (std::size_t k = 0; k < K; ++k) {
    for (std::size_t v = 0; v < V; ++v) {
      phi_kv[k * V + v] = (n_vk[v * K + k] + beta) / (n_k[k] + K * beta);
    }
  }
}

int main(int argc, char* argv[])
{
  // 人工的にデータをつくる
  std::size_t const M =  20; // 文章数
  std::size_t const V = 100; // 語彙数
  std::size_t const K =   4; // トピック数

  double const alpha = 0.1;
  double const beta = 2.0;

  std::mt19937_64 engine(std::random_device("")());

  std::vector<std::size_t> w(1000); // 単語
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
      std::size_t k = 0;
      for (double const xi = std::uniform_real_distribution<double>(0.0, theta[m * (K + 1) + K])(engine); xi > theta[m * (K + 1) + (k + 1)]; ++k);

      // k-th トピックの単語分布から単語 v をサンプリング
      std::size_t v = 0;
      for (double const xi = std::uniform_real_distribution<double>(0.0, phi[k * (V + 1) + V])(engine); xi > phi[k * (V + 1) + (v + 1)]; ++v);

      w[i] = v;
    }
  }

  lda(w, n, M, V, K, alpha, beta, 100, 1000);

  return 0;
}
