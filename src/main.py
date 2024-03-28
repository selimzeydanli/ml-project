from src import orderdb, truckdb, assignment, transaction, plotting


def main():
    orderdb.run()
    truckdb.run()
    assignment.run()
    transaction.run()
    plotting.run()

if __name__ == "__main__":
    main()
