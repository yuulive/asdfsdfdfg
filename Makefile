RUNEXAMPLE = cargo run --example

examples = bode linear_system oscillation polar poly transfer_function

.PHONY : doc $(examples)

# Create documentation without dependencies.
doc:
	cargo doc --no-deps

# Run all examples
all_examples: $(examples)

# '$@' is the name of the target
$(examples):
	$(RUNEXAMPLE) $@
