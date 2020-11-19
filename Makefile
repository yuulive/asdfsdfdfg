RUNEXAMPLE = cargo run --example

examples = arma_channel \
		   bode \
		   discrete \
		   discretization \
		   linear_system \
		   oscillation \
		   polar \
		   poly \
		   root_locus \
		   suspension \
		   transfer_function

.PHONY : all_examples clippy doc $(examples) html

# Create documentation without dependencies.
doc:
	cargo doc --no-deps

# Build html documentation from markdown files.
html:
	cd design/ && ./build.sh

# Clippy linting for code, tests and examples with pedantic lints
clippy:
	cargo clippy --all-targets -- -W clippy::pedantic

# Run all examples
all_examples: $(examples)

# '$@' is the name of the target
$(examples):
	$(RUNEXAMPLE) $@
