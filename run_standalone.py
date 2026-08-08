import predictions_code

if __name__ == "__main__":
    labels = predictions_code.main()

    with open("result.csv", "w") as f:
        for p in labels:
            f.write(str(p) + "\n")

    print("result.csv. created successfully!")
