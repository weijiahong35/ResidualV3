

def order_for_epoch(count, seed, epoch):
    import random
    order=list(range(count));random.Random(seed+epoch).shuffle(order)
    return order
