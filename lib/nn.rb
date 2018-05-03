require "numo/narray"
require "json"

class NN
  VERSION = "1.8"

  include Numo

  attr_accessor :weights
  attr_accessor :biases
  attr_accessor :gammas
  attr_accessor :betas
  attr_accessor :learning_rate
  attr_accessor :batch_size
  attr_accessor :activation
  attr_accessor :momentum
  attr_accessor :weight_decay
  attr_accessor :dropout_ratio
  attr_reader :training

  def initialize(num_nodes,
                 learning_rate: 0.01,
                 batch_size: 1,
                 activation: %i(relu identity),
                 momentum: 0,
                 weight_decay: 0,
                 use_dropout: false,
                 dropout_ratio: 0.5,
                 use_batch_norm: false)
    SFloat.srand(rand(2 ** 64))
    @num_nodes = num_nodes
    @learning_rate = learning_rate
    @batch_size = batch_size
    @activation = activation
    @momentum = momentum
    @weight_decay = weight_decay
    @use_dropout = use_dropout
    @dropout_ratio = dropout_ratio
    @use_batch_norm = use_batch_norm
    init_weight_and_bias
    init_gamma_and_beta if @use_batch_norm
    @training = true
    init_layers
  end

  def self.load(file_name)
    json = JSON.parse(File.read(file_name))
    nn = self.new(json["num_nodes"],
      learning_rate: json["learning_rate"],
      batch_size: json["batch_size"],
      activation: json["activation"].map(&:to_sym),
      momentum: json["momentum"],
      weight_decay: json["weight_decay"],
      use_dropout: json["use_dropout"],
      dropout_ratio: json["dropout_ratio"],
      use_batch_norm: json["use_batch_norm"],
    )
    nn.weights = json["weights"].map{|weight| SFloat.cast(weight)}
    nn.biases = json["biases"].map{|bias| SFloat.cast(bias)}
    if json["use_batch_norm"]
      nn.gammas = json["gammas"].map{|gamma| SFloat.cast(gamma)}
      nn.betas = json["betas"].map{|beta| SFloat.cast(beta)}
    end
    nn
  end

  def train(x_train, y_train, epochs,
            learning_rate_decay: 0,
            save_dir: nil,
            save_interval: 1,
            test: nil,
            border: nil,
            tolerance: 0.5,
            &block)
    num_train_data = x_train.is_a?(SFloat) ? x_train.shape[0] : x_train.length
    (1..epochs).each do |epoch|
      loss = nil
      (num_train_data.to_f / @batch_size).ceil.times do
        loss = learn(x_train, y_train, &block)
        if loss.nan?
          puts "loss is nan"
          return
        end
      end
      if save_dir && epoch % save_interval == 0
        save("#{save_dir}/epoch#{epoch}.json")
      end
      msg = "epoch #{epoch}/#{epochs} loss: #{loss}"
      if test
        acc = accurate(*test, tolerance, &block)
        puts "#{msg} accurate: #{acc}"
        break if border && acc >= border
      else
        puts msg
      end
      @learning_rate -= learning_rate_decay
      @learning_rate = 1e-7 if @learning_rate < 1e-7
    end
  end

  def test(x_test, y_test, tolerance = 0.5, &block)
    acc = accurate(x_test, y_test, tolerance, &block)
    puts "accurate: #{acc}"
    acc
  end

  def accurate(x_test, y_test, tolerance = 0.5, &block)
    correct = 0
    num_test_data = x_test.is_a?(SFloat) ? x_test.shape[0] : x_test.length
    (num_test_data.to_f / @batch_size).ceil.times do |i|
      x = SFloat.zeros(@batch_size, @num_nodes.first)
      y = SFloat.zeros(@batch_size, @num_nodes.last)
      @batch_size.times do |j|
        k = i * @batch_size + j
        break if k >= num_test_data
        if x_test.is_a?(SFloat)
          x[j, true] = x_test[k, true]
          y[j, true] = y_test[k, true]
        else
          x[j, true] = SFloat.cast(x_test[k])
          y[j, true] = SFloat.cast(y_test[k])
        end
      end
      x, y = block.call(x, y) if block
      out = forward(x, false)
      @batch_size.times do |j|
        vout = out[j, true]
        vy = y[j, true]
        case @activation[1]
        when :identity
          correct += 1 unless (NMath.sqrt((vout - vy) ** 2) < tolerance).to_a.include?(0)
        when :softmax
          correct += 1 if vout.max_index == vy.max_index
        end
      end
    end
    correct.to_f / num_test_data
  end

  def learn(x_train, y_train, &block)
    x = SFloat.zeros(@batch_size, @num_nodes.first)
    y = SFloat.zeros(@batch_size, @num_nodes.last)
    @batch_size.times do |i|
      if x_train.is_a?(SFloat)
        r = rand(x_train.shape[0])
        x[i, true] = x_train[r, true]
        y[i, true] = y_train[r, true]
      else
        r = rand(x_train.length)
        x[i, true] = SFloat.cast(x_train[r])
        y[i, true] = SFloat.cast(y_train[r])
      end
    end
    x, y = block.call(x, y) if block
    forward(x)
    backward(y)
    update_weight_and_bias
    update_gamma_and_beta if @use_batch_norm
    @layers[-1].loss(y)
  end

  def run(x)
    x = SFloat.cast(x) if x.is_a?(Array)
    out = forward(x, false)
    out.to_a
  end

  def save(file_name)
    json = {
      "version" => VERSION,
      "num_nodes" => @num_nodes,
      "learning_rate" => @learning_rate,
      "batch_size" => @batch_size,
      "activation" => @activation,
      "momentum" => @momentum,
      "weight_decay" => @weight_decay,
      "use_dropout" => @use_dropout,
      "dropout_ratio" => @dropout_ratio,
      "use_batch_norm" => @use_batch_norm,
      "weights" => @weights.map(&:to_a),
      "biases" => @biases.map(&:to_a),
    }
    if @use_batch_norm
      json_batch_norm = {
        "gammas" => @gammas,
        "betas" => @betas
      }
      json.merge!(json_batch_norm)
    end
    File.write(file_name, JSON.dump(json))
  end

  private

  def init_weight_and_bias
    @weights = Array.new(@num_nodes.length - 1)
    @biases = Array.new(@num_nodes.length - 1)
    @weight_amounts = Array.new(@num_nodes.length - 1, 0)
    @bias_amounts = Array.new(@num_nodes.length - 1, 0)
    @num_nodes[0...-1].each_index do |i|
      weight = SFloat.new(@num_nodes[i], @num_nodes[i + 1]).rand_norm
      bias = SFloat.new(@num_nodes[i + 1]).rand_norm
      if @activation[0] == :relu
        @weights[i] = weight / Math.sqrt(@num_nodes[i]) * Math.sqrt(2)
        @biases[i] = bias / Math.sqrt(@num_nodes[i]) * Math.sqrt(2)
      else
        @weights[i] = weight / Math.sqrt(@num_nodes[i])
        @biases[i] = bias / Math.sqrt(@num_nodes[i])
      end
    end
  end

  def init_gamma_and_beta
    @gammas = Array.new(@num_nodes.length - 2, 1)
    @betas = Array.new(@num_nodes.length - 2, 0)
    @gamma_amounts = Array.new(@num_nodes.length - 2, 0)
    @beta_amounts = Array.new(@num_nodes.length - 2, 0)
  end

  def init_layers
    @layers = []
    @num_nodes[0...-2].each_index do |i|
      @layers << Affine.new(self, i)
      @layers << BatchNorm.new(self, i) if @use_batch_norm
      @layers << case @activation[0]
      when :sigmoid
        Sigmoid.new
      when :relu
        ReLU.new
      end
      @layers << Dropout.new(self) if @use_dropout
    end
    @layers << Affine.new(self, -1)
    @layers << case @activation[1]
    when :identity
      Identity.new(self)
    when :softmax
      Softmax.new(self)
    end
  end

  def forward(x, training = true)
    @training = training
    @layers.each do |layer|
      x = layer.forward(x)
    end
    x
  end

  def backward(y)
    dout = @layers[-1].backward(y)
    @layers[0...-1].reverse.each do |layer|
      dout = layer.backward(dout)
    end
  end

  def update_weight_and_bias
    @layers.select{|layer| layer.is_a?(Affine)}.each.with_index do |layer, i|
      weight_amount = layer.d_weight.mean(0) * @learning_rate
      @weight_amounts[i] = weight_amount + @momentum * @weight_amounts[i]
      @weights[i] -= @weight_amounts[i]
      bias_amount = layer.d_bias.mean * @learning_rate
      @bias_amounts[i] = bias_amount + @momentum * @bias_amounts[i]
      @biases[i] -= @bias_amounts[i]
    end
  end

  def update_gamma_and_beta
    @layers.select{|layer| layer.is_a?(BatchNorm)}.each.with_index do |layer, i|
      gamma_amount = layer.d_gamma.mean * @learning_rate
      @gamma_amounts[i] = gamma_amount + @momentum * @gamma_amounts[i]
      @gammas[i] -= @gamma_amounts[i]
      beta_amount = layer.d_beta.mean * @learning_rate
      @beta_amounts[i] = beta_amount + @momentum * @beta_amounts[i]
      @betas[i] -= @beta_amounts[i]
    end
  end
end


class NN::Affine
  include Numo

  attr_reader :d_weight
  attr_reader :d_bias

  def initialize(nn, index)
    @nn = nn
    @index = index
    @d_weight = nil
    @d_bias = nil
  end

  def forward(x)
    @x = x
    @x.dot(@nn.weights[@index]) + @nn.biases[@index]
  end

  def backward(dout)
    x = @x.reshape(*@x.shape, 1)
    d_ridge = @nn.weight_decay * @nn.weights[@index]
    @d_weight = x.dot(dout.reshape(dout.shape[0], 1, dout.shape[1])) + d_ridge
    @d_bias = dout
    dout.dot(@nn.weights[@index].transpose)
  end
end


class NN::Sigmoid
  def forward(x)
    @out = 1.0 / (1 + Numo::NMath.exp(-x))
  end

  def backward(dout)
    dout * (1.0 - @out) * @out
  end
end


class NN::ReLU
  def forward(x)
    @x = x.clone
    x[x < 0] = 0
    x
  end

  def backward(dout)
    @x[@x > 0] = 1.0
    @x[@x <= 0] = 0.0
    dout * @x
  end
end


class NN::Identity
  include Numo

  def initialize(nn)
    @nn = nn
  end

  def forward(x)
    @out = x
  end

  def backward(y)
    @out - y
  end

  def loss(y)
    ridge = 0.5 * @nn.weight_decay * @nn.weights.reduce(0){|sum, weight| sum + (weight ** 2).sum}
    0.5 * ((@out - y) ** 2).sum / @nn.batch_size + ridge
  end
end


class NN::Softmax
  include Numo

  def initialize(nn)
    @nn = nn
  end

  def forward(x)
    @out = NMath.exp(x) / NMath.exp(x).sum(1).reshape(x.shape[0], 1)
  end

  def backward(y)
    @out - y
  end

  def loss(y)
    ridge = 0.5 * @nn.weight_decay * @nn.weights.reduce(0){|sum, weight| sum + (weight ** 2).sum}
    -(y * NMath.log(@out + 1e-7)).sum / @nn.batch_size + ridge
  end
end


class NN::Dropout
  include Numo

  def initialize(nn)
    @nn = nn
    @mask = nil
  end

  def forward(x)
    if @nn.training
      @mask = SFloat.ones(*x.shape).rand < @nn.dropout_ratio
      x[@mask] = 0
    else
      x *= (1 - @nn.dropout_ratio)
    end
    x
  end

  def backward(dout)
    dout[@mask] = 0 if @nn.training
    dout
  end
end


class NN::BatchNorm
  include Numo

  attr_reader :d_gamma
  attr_reader :d_beta

  def initialize(nn, index)
    @nn = nn
    @index = index
  end

  def forward(x)
    @x = x
    @mean = x.mean(0)
    @xc = x - @mean
    @var = (@xc ** 2).mean(0)
    @std = NMath.sqrt(@var + 1e-7)
    @xn = @xc / @std
    out = @nn.gammas[@index] * @xn + @nn.betas[@index]
    out.reshape(*@x.shape)
  end

  def backward(dout)
    @d_beta = dout.sum(0)
    @d_gamma = (@xn * dout).sum(0)
    dxn = @nn.gammas[@index] * dout
    dxc = dxn / @std
    dstd = -((dxn * @xc) / (@std ** 2)).sum(0)
    dvar = 0.5 * dstd / @std
    dxc += (2.0 / @nn.batch_size) * @xc * dvar
    dmean = dxc.sum(0)
    dx = dxc - dmean / @nn.batch_size
    dx.reshape(*@x.shape)
  end
end
