module CIFAR10
  def self.load_train(index)
    if File.exist?("CIFAR-10-train#{index}.marshal")
      marshal = File.binread("CIFAR-10-train#{index}.marshal")
      return Marshal.load(marshal)
    end
    bin = File.binread("#{dir}/data_batch_#{index}.bin")
    datasets = bin.unpack("C*")
    x_train = []
    y_train = []
    loop do
      label = datasets.shift
      break unless label
      x_train << datasets.slice!(0, 3072)
      y_train << label
    end
    train = [x_train, y_train]
    File.binwrite("CIFAR-10-train#{index}.marshal", Marshal.dump(train))
    train
  end

  def self.load_test
    if File.exist?("CIFAR-10-test.marshal")
      marshal = File.binread("CIFAR-10-test.marshal")
      return Marshal.load(marshal)
    end
    bin = File.binread("#{dir}/test_batch.bin")
    datasets = bin.unpack("C*")
    x_test = []
    y_test = []
    loop do
      label = datasets.shift
      break unless label
      x_test << datasets.slice!(0, 3072)
      y_test << label
    end
    test = [x_test, y_test]
    File.binwrite("CIFAR-10-test.marshal", Marshal.dump(test))
    test
  end

  def self.categorical(y_data)
    y_data = y_data.map do |label|
      classes = Array.new(10, 0)
      classes[label] = 1
      classes
    end
  end

  def self.dir
    "cifar-10-batches-bin"
  end
end
