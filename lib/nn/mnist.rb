require "zlib"

module MNIST
  def self.load_train
    if File.exist?("mnist/train.marshal")
      marshal = File.binread("mnist/train.marshal")
      Marshal.load(marshal)
    else
      x_train, y_train = load("mnist/train-images-idx3-ubyte.gz", "mnist/train-labels-idx1-ubyte.gz")
      marshal = Marshal.dump([x_train, y_train])
      File.binwrite("mnist/train.marshal", marshal)
      [x_train, y_train]
    end
  end

  def self.load_test
    if File.exist?("mnist/test.marshal")
      marshal = File.binread("mnist/test.marshal")
      Marshal.load(marshal)
    else
      x_test, y_test = load("mnist/t10k-images-idx3-ubyte.gz", "mnist/t10k-labels-idx1-ubyte.gz")
      marshal = Marshal.dump([x_test, y_test])
      File.binwrite("mnist/test.marshal", marshal)
      [x_test, y_test]
    end
  end

  def self.categorical(y_data)
    y_data = y_data.map do |label|
      classes = Array.new(10, 0)
      classes[label] = 1
      classes
    end
  end

  private_class_method

  def self.load(images_file_name, labels_file_name)
    images = []
    labels = nil
    Zlib::GzipReader.open(images_file_name) do |f|
      magic, n_images = f.read(8).unpack("N2")
      n_rows, n_cols = f.read(8).unpack("N2")
      n_images.times do
        images << f.read(n_rows * n_cols).unpack("C*")
      end
    end
    Zlib::GzipReader.open(labels_file_name) do |f|
      magic, n_labels = f.read(8).unpack("N2")
      labels = f.read(n_labels).unpack("C*")
    end
    [images, labels]
  end
end
