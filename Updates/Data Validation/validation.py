import time

if file_name.endswith("_v1"):
    expectation_suite.expect_column_values_to_be_in_set(
        column="threshold",
        value_set=[0]
    )
if file_name.endswith("_v2"):
    expectation_suite.expect_column_values_to_be_unique(
        column="threshold"
    )
expectation_suite.expect_column_pair_values_A_to_be_greater_than_or_equal_to_B(
    column_A="filtered_height",
    column_B="threshold"
)

expectation_suite.expect_column_values_to_be_equal(
    column="toy_status",
    value=1
)
expectation_suite.expect_column_values_to_be_equal(
    column="toy_status",
    value=1
)

expectation_suite.expect_column_pair_values_A_to_be_greater_than_or_equal_to_B(
    column_A="filtered_height",
    column_B="threshold"
)

def expect_toy_status_to_alternate_every_15_seconds(context, batch_data):
    start_time = time.time()
    alternating_status = 0

    for index, row in batch_data.iterrows():
        current_time = time.time()
        elapsed_time = current_time - start_time

        # Switch toy_status every 15 seconds
        if elapsed_time >= 15:
            alternating_status = 1 if alternating_status == 0 else 0
            start_time = current_time  # Reset timer

        if row['toy_status'] != alternating_status:
            return False  # Failure if not alternating as expected

    return True  

 if file_name.endswith("_v2"):
    expectation_suite.add_expectation(
        custom_expectation=expect_toy_status_to_alternate_every_15_seconds
    )
