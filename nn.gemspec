
lib = File.expand_path("../lib", __FILE__)
$LOAD_PATH.unshift(lib) unless $LOAD_PATH.include?(lib)
require "nn"

Gem::Specification.new do |spec|
  spec.name          = "nn"
  spec.version       = NN::VERSION + ".1"
  spec.authors       = ["unagiootoro"]
  spec.email         = ["ootoro838861@outlook.jp"]

  spec.summary       = %q{Ruby用ニューラルネットワークライブラリ}
  spec.description   = %q{Rubyでニューラルネットワークを作成できます。}
  spec.homepage      = "https://github.com/unagiootoro/nn.git"
  spec.license       = "MIT"

  spec.add_dependency "numo-narray"

  # Prevent pushing this gem to RubyGems.org. To allow pushes either set the 'allowed_push_host'
  # to allow pushing to a single host or delete this section to allow pushing to any host.
=begin
  if spec.respond_to?(:metadata)
    spec.metadata["allowed_push_host"] = "TODO: Set to 'http://mygemserver.com'"
  else
    raise "RubyGems 2.0 or newer is required to protect against " \
      "public gem pushes."
  end
=end

  spec.files         = `git ls-files -z`.split("\x0").reject do |f|
    f.match(%r{^(test|spec|features)/})
  end
  spec.bindir        = "exe"
  spec.executables   = spec.files.grep(%r{^exe/}) { |f| File.basename(f) }
  spec.require_paths = ["lib"]

  spec.add_development_dependency "bundler", "~> 1.16"
  spec.add_development_dependency "rake", "~> 10.0"
end
