RUNEXAMPLE = cargo run --example

examples = arma_channel \
		   bode \
		   discrete \
		   discretization \
		   epidemic \
		   linear_system \
		   oscillation \
		   polar \
		   poly \
		   root_locus \
		   suspension \
		   transfer_function

.PHONY : all all_examples check-format clippy doc $(examples) html update-version

# Run build, tests and examples, in debug mode
all:
	cargo c && cargo t && make all_examples

# Run all examples
all_examples: $(examples)

# Check code format, does not apply changes
check-format:
	cargo fmt --all -- --check

# Clippy linting for code, tests and examples with pedantic lints
clippy:
	cargo clippy --all-targets -- -W clippy::pedantic

# Create documentation without dependencies.
doc:
	cargo doc --no-deps

# '$@' is the name of the target
$(examples):
	$(RUNEXAMPLE) $@

# Build html documentation from markdown files.
html:
	cd design/ && ./build.sh

# Update version
# run as:
#    make update-version VERSION=0.10.0
update-version:
	./update-version.sh $(VERSION)
