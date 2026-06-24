logistic_accuracy = 65.39
mlp_accuracy = 62.78

print("\nMODEL COMPARISON\n")

print(
    f"Logistic Regression: {logistic_accuracy:.2f}%"
)

print(
    f"MLP Classifier: {mlp_accuracy:.2f}%"
)

if logistic_accuracy > mlp_accuracy:

    print(
        "\nSelected Model: Logistic Regression"
    )

else:

    print(
        "\nSelected Model: MLP"
    )