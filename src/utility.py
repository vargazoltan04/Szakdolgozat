#Megkeresi a minimumpontokat egy tömbben (csak akkor találja meg, ha azok 0-k)
#ha több van közvetlen egymás mellett, akkor a legutolsó pontot találja meg

def find_local_minimum_points(arr):
    arr_minimum_points = []

    for i in range(len(arr) - 1):
        if arr[i] != 0 and arr[i+1] == 0:
            arr_minimum_points.append(i)

    return arr_minimum_points