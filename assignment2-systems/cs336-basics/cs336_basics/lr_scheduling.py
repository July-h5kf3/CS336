import math

def lr_cosine_schedule(t,a_max,a_min,t_w,t_c):
    if t < t_w:
        return t / t_w * a_max
    elif t <= t_c:
        return a_min + (a_max - a_min) * (1 + math.cos(math.pi * (t - t_w) / (t_c - t_w))) / 2
    else:
        return a_min