RUNEXAMPLE = cargo run --example

examples = bode \
		   discrete \
		   discretization \
		   linear_system \
		   oscillation \
		   polar \
		   poly \
		   transfer_function

.PHONY : all_examples clippy doc $(examples)

# Create documentation without dependencies.
doc:
	cargo doc --no-deps

# Clippy linting for code, tests and examples with pedantic lints
clippy:
	cargo clippy --all-targets -- -W clippy::pedantic

# Run all examples
all_examples: $(examples)

# '$@' is the name of the target
$(examples):
	$(RUNEXAMPLE) $@
