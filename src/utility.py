def find_local_minimum_points(arr):
    arr_minimum_points = []

    for i in range(len(arr) - 1):
        if arr[i] != 0 and arr[i+1] == 0:
            arr_minimum_points.append(i)

    print(arr_minimum_points)
    return arr_minimum_points