def find_error_position(filename, position):
    with open(filename, 'rb') as f:  # Open in binary mode
        f.seek(position - 50)  # Go to 50 bytes before the error position
        data = f.read(100)  # Read 100 bytes (50 bytes before and after)
        print(data.decode('ISO-8859-1'))  # Decode using 'ISO-8859-1' to avoid further decode errors

error_position = 259712
find_error_position('data/single_cell.csv', error_position)


