from my_utils.calculate_hamming_distances import hamming_distance_sum

input_list = [[0,1,2,3,4],[1,2,2,3,4],[0,1,2,2,2],[1,1,2,3,2],[7,7,7,7,7]]

hamming_distance = hamming_distance_sum(input_list)

print(hamming_distance)