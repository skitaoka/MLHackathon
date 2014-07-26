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

// 決定木のデータ構造
namespace
{
  template <typename Data>
  struct decision_tree
  {
    virtual int operator () (Data const & data) const = 0;
  };

  template <typename Data>
  struct dt_node : public decision_tree<Data>
  {
    dt_node(int const question, std::vector<std::shared_ptr<decision_tree<Data>>> children)
      : question_(question)
      , children_(children)
    {
    }

    virtual int operator () (Data const & data) const
    {
      return (*children_[data(question_)])(data);
    }

  private:
    int const question_; // 質問番号

    std::vector<std::shared_ptr<decision_tree<Data>>> children_;
  };

  template <typename Data>
  struct dt_leaf : public decision_tree<Data>
  {
    dt_leaf(int const klass)
      : klass_(klass)
    {
    }

    virtual int operator () (Data const & data) const
    {
      return klass_;
    }

  private:
    int const klass_; // 答えのクラス
  };

  template <typename Data>
  double entropy_function(int const question, std::size_t const i, std::vector<Data> const & data)
  {
    // 質問 question について、答えが i になる要素についてのクラスの分布からエントロピーを求める
    std::vector<double> p(iris_data::num_classes, 0);
    for (Data const & d : data) {
      if (d(question) == i) {
        ++p[d.klass()];
      }
    }
    double const count = std::accumulate(std::begin(p), std::end(p), 0.0);
    if (count <= 0) {
      return 0.0; // 要素が空のとき、エントロピー関数の値は 0
    }
    std::transform(std::begin(p), std::end(p), std::begin(p),
      std::bind1st(std::multiplies<double>(), 1.0 / count));

    double retval
#if 1
      // エントロピー
      = 0.0;
    for (std::size_t i = 0; i < iris_data::num_classes; ++i) {
      if (p[i] > 0.0) {
        retval -= p[i] * std::log(p[i]);
      }
    }
#else
      // ジニ係数
      = 1.0;
    for (std::size_t i = 0; i < iris_data::num_classes; ++i) {
      retval -= p[i] * p[i];
    }
#endif
    return retval;
  }

  // 情報ゲインを計算する
  template <typename Data>
  double find_information_gain(int const question, std::vector<Data> const & data)
  {
    std::vector<int> frequency(iris_data::num_dimensions, 0);
    for (Data const & datum : data) {
      ++frequency[datum(question)];
    }

    // ノード内で第一項は変化しないので計算しない
    double retval = 0.0;
    for (std::size_t i = 0; i < iris_data::num_dimensions; ++i) {
      retval -= frequency[i] * entropy_function(question, i, data);
    }
    return retval / data.size();
  }

  // 決定木を構築する
  template <typename Data>
  std::shared_ptr<decision_tree<Data>>
    build_decision_tree(
    std::vector<int > const & questions,
    std::vector<Data> const & data)
  {
    // データが無いときはデフォルト値: 0
    if (data.empty()) {
      return std::shared_ptr<decision_tree<Data>>(new dt_leaf<Data>(0));
    }

    // 質問が空のときは多数決で
    if (questions.empty()) {
      std::vector<std::size_t> bin(iris_data::num_classes, 0);
      for (Data const & datum : data) {
        ++bin[datum.klass()];
      }
      auto it = std::max_element(std::begin(bin), std::end(bin));
      int const klass = static_cast<int>(std::distance(std::begin(bin), it));
      return std::shared_ptr<decision_tree<Data>>(new dt_leaf<Data>(klass));
    }

    // すでに全て同じクラスならそのクラスで
    if (std::all_of(std::begin(data), std::end(data), [&data](Data const & d){ return data.front().klass() == d.klass(); })) {
      return std::shared_ptr<decision_tree<Data>>(new dt_leaf<Data>(data.front().klass()));
    }

    // 全ての質問を試す
    double information_gain_max = -std::numeric_limits<double>::infinity();
    int good_question = 0;
    for (int question : questions) {
      double const information_gain = find_information_gain(question, data);
      if (information_gain_max < information_gain) {
        information_gain_max = information_gain;
        good_question = question;
      }
    }

    // 一番よかった質問で、再帰的につくる
    std::vector<std::shared_ptr<decision_tree<Data>>>
      children(Data::num_dimensions); // 子ノードの集合

    // 採用した質問を除いた質問の集合をつくる
    std::vector<int> new_questions;
    for (int const question : questions) {
      if (question != good_question) {
        new_questions.push_back(question);
      }
    }

    for (std::size_t i = 0; i < Data::num_dimensions; ++i) {
      std::vector<Data> new_data;
      for (Data const & d : data) {
        if (i == d(good_question)) {
          new_data.push_back(d); // good_question に i とだけ答えるデータの集合を新規データとしてつくる
        }
      }

      children[i] = build_decision_tree(new_questions, new_data);
    }
    return std::shared_ptr<decision_tree<Data>>(new dt_node<Data>(good_question, std::move(children)));
  }
}

// ユーティリティ
namespace
{
  // 連続値を離散変数に変換する
  template <std::size_t N>
  std::vector<int> d2i(std::vector<double> data)
  {
    std::vector<double> sorted_data(data);
    std::sort(std::begin(sorted_data), std::end(sorted_data));

    // 適当に区切る
    std::vector<double> threshold(N);
    for (std::size_t i = 1; i < N; ++i) {
      threshold[i - 1] = sorted_data[sorted_data.size() * i / N];
    }
    threshold[N - 1] = sorted_data.back();

    // 変換
    std::vector<int> retval(data.size());
    std::transform(std::begin(data), std::end(data),
      std::begin(retval), [&](double const x) -> int
    {
      int dim = 0;
      while (x > threshold[dim]) {
        ++dim;
      }
      return dim;
    });
    return retval;
  }
}

// データ
namespace
{
  struct iris_data
  {
    static std::size_t const num_classes = 3; // 分類するクラスの数
    static std::size_t const num_questions = 4; // 特徴変数の数
    static std::size_t const num_dimensions = 8; // 連続変数の離散化解像度

    iris_data(std::vector<int> answers, int const klass)
      : answers_(std::move(answers))
      , klass_(klass)
    {
    }

    iris_data(iris_data const & rhs)
      : answers_(rhs.answers_)
      , klass_(rhs.klass_)
    {
    }

    // question 番目の質問の答えを返す
    inline int operator () (int const question) const
    {
      return answers_[question];
    }

    inline int klass() const
    {
      return klass_;
    }

    std::vector<int> const answers_;
    int              const klass_;
  };
}

int main(int argc, char* argv[])
{
  std::vector<iris_data> data;
  {
    std::vector<double> q1d;
    std::vector<double> q2d;
    std::vector<double> q3d;
    std::vector<double> q4d;
    std::vector<int   > ans;

    // データ読み込み
    {
      std::map<std::string, int> t2i;
      t2i["setosa"] = 0;
      t2i["versicolor"] = 1;
      t2i["virginica"] = 2;

      std::ifstream in("iris.dat");
      if (!in) {
        std::clog << "Could not open file: " << argv[1] << '\n';
        return 1;
      }
      while (in) {
        int id;
        double q1, q2, q3, q4;
        std::string tag;

        in >> id >> q1 >> q2 >> q3 >> q4 >> tag;
        if (tag.empty()) {
          break;
        }

        q1d.push_back(q1);
        q2d.push_back(q2);
        q3d.push_back(q3);
        q4d.push_back(q4);
        ans.push_back(t2i.find(tag)->second);
      }
    }

    // 離散変数に変換
    std::vector<int> q1i = d2i<iris_data::num_dimensions>(q1d);
    std::vector<int> q2i = d2i<iris_data::num_dimensions>(q2d);
    std::vector<int> q3i = d2i<iris_data::num_dimensions>(q3d);
    std::vector<int> q4i = d2i<iris_data::num_dimensions>(q4d);

    for (std::size_t i = 0, size = ans.size(); i < size; ++i) {
      std::vector<int> q(4);
      q[0] = q1i[i];
      q[1] = q2i[i];
      q[2] = q3i[i];
      q[3] = q4i[i];
      data.push_back(iris_data(std::move(q), ans[i]));
    }
  }

  // 決定木の個数
  std::size_t const num_decision_trees = 100;

  // 学習に利用する質問の数
  // 分類なら sqrt(num_questions), 回帰なら num_questions/3) ぐらいがいいらしい
  // cf. http://ja.wikipedia.org/wiki/Random_forest
  std::size_t const num_learn_questions = iris_data::num_questions / 2;

  // 学習に利用するデータの数
  std::size_t const num_sample_size = data.size() * 2 / 3;

  std::vector<std::size_t> questions(iris_data::num_questions);
  std::iota(std::begin(questions), std::end(questions), 0);

  std::vector<std::size_t> indices(data.size());
  std::iota(std::begin(indices), std::end(indices), 0);

  // 決定木 (の集合)
  std::vector<std::shared_ptr<decision_tree<iris_data>>> trees;
  std::mt19937_64 engine(std::random_device("")());
  for (std::size_t k = 0; k < num_decision_trees; ++k) {
    // 質問の集合をランダムにつくる
    std::shuffle(std::begin(questions), std::end(questions), engine);
    std::vector<int> questions(std::begin(questions), std::begin(questions) + num_learn_questions);

    // データの集合をランダムにつくる
    std::shuffle(std::begin(indices), std::end(indices), engine);
    std::vector<iris_data> learn_data;
    learn_data.reserve(num_sample_size);
    std::transform(std::begin(indices), std::begin(indices) + num_sample_size,
      std::back_inserter(learn_data), [&data](int const i) { return data[i]; });

    trees.push_back(build_decision_tree(questions, learn_data));
  }

  // 全て判定してみる
  int num_good = 0; // 正解数
  for (std::size_t i = 0, size = data.size(); i < size; ++i) {
    std::vector<std::size_t> bin(iris_data::num_classes, 0);
    for (std::size_t k = 0; k < num_decision_trees; ++k) {
      ++bin[(*trees[k])(data[i])];
    }

    std::cout << (i + 1) << '\t';
    for (std::size_t j = 0; j < iris_data::num_classes; ++j) {
      for (std::size_t n = 0, count = bin[j]; n < count; ++n) {
        std::cout << j;
      }
    }
    std::cout << '\n';

    // 答え
    auto it = std::max_element(std::begin(bin), std::end(bin));
    int const klass = static_cast<int>(std::distance(std::begin(bin), it));
    num_good += (data[i].klass() == klass);
  }

  // 正答数
  std::cout << num_good << '/' << data.size() << '\n';

  return 0;
}
