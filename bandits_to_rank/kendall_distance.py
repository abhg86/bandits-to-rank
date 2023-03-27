def kendall_distance(list1, list2):
    if len(list1) != len(list2):
        raise ValueError('Lists must have the same length')
    
    new_order_list2 = [list1.index(nb) for nb in list2]
    
    kendall_count = 0
    for i in range(len(list1)):
        for j in range(i+1, len(list1)):
            if (new_order_list2[i] > new_order_list2[j]):
                kendall_count += 1
    
    return kendall_count