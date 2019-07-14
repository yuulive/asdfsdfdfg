RUNEXAMPLE = cargo run --example

.PHONY : doc examples bode linear_system oscillation poly transfer_function

# Create documentation without dependencies.
doc:
	cargo doc --no-deps

# Run all examples
examples: bode linear_system oscillation poly transfer_function

bode:
	$(RUNEXAMPLE) bode

linear_system:
	$(RUNEXAMPLE) linear_system

oscillation:
	$(RUNEXAMPLE) oscillation

poly:
	$(RUNEXAMPLE) poly

transfer_function:
	$(RUNEXAMPLE) transfer_function
