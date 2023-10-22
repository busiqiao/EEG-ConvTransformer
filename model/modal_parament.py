def select_modal(num_class=6, variant=0):
    model_param = ModelParam()
    if num_class == 6:
        if variant == 0:
            model_param = ModelParam(num_class=6, num_heads=4, channels=8, E=16, F=256, T=32, gamma=0.5, depth=2,
                                     decay=0.135, epochs=35, seed_value=3407)
        elif variant == 1:
            model_param = ModelParam(num_class=6, num_heads=8, channels=32, E=128, F=512, T=32, gamma=0.5, depth=2,
                                     decay=0.180, epochs=35, seed_value=3407)
        elif variant == 2:
            model_param = ModelParam(num_class=6, num_heads=12, channels=72, E=432, F=768, T=32, gamma=0.6, depth=2,
                                     decay=0.170, epochs=40, seed_value=3407)
        else:
            print('Not modal！')
    elif num_class == 72:
        if variant == 0:
            model_param = ModelParam(num_class=72, num_heads=4, channels=8, E=16, F=256, T=32, gamma=0.7, depth=2,
                                     decay=0.016, epochs=80, seed_value=3407)
        elif variant == 1:
            model_param = ModelParam(num_class=72, num_heads=8, channels=32, E=128, F=512, T=32, gamma=0.7, depth=2,
                                     decay=0.03, epochs=35, seed_value=3407)
        elif variant == 2:
            model_param = ModelParam(num_class=72, num_heads=12, channels=72, E=432, F=768, T=32, gamma=0.7, depth=2,
                                     decay=0.035, epochs=35, seed_value=3407)
        else:
            print('Not modal！')

    return model_param


class ModelParam:
    def __init__(self, num_class=6, num_heads=4, channels=8, E=16, F=256, T=32, gamma=0.5, depth=2, decay=0.135,
                 epochs=35, seed_value=3407):
        self.num_class = num_class
        self.num_heads = num_heads
        self.channels = channels
        self.E = E
        self.F = F
        self.T = T
        self.depth = depth
        self.decay = decay
        self.gamma = gamma
        self.epochs = epochs
        self.seed_value = seed_value
