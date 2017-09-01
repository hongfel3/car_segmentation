def adjust_learning_rate(optimizer, epoch, args_lr, epoch_adjust):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args_lr * (0.1 ** (epoch // epoch_adjust))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
