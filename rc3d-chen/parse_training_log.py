import numpy as np

with open('metatrain_bug_fixed.log') as input, open('metatrain_bug_fixed_cross_ent.log', 'w') as cross_ent_out, \
    open('metatrain_bug_fixed_losses.log', 'w') as losses_out:
    cross_ent_buffer = []
    for l in input.readlines():
        if 'Cross entropy loss' in l:
            cross_ent_buffer.append(float(l.split(' ')[-1]))
            if len(cross_ent_buffer) == 10:
                cross_ent_out.write(str(np.mean(cross_ent_buffer)) + '\n')
                cross_ent_buffer = []
        if 'rpn_cls' in l:
            split = l.split(' ')
            losses_out.write(split[1] + split[3] + split[5] + split[7])
