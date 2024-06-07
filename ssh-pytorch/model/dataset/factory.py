from model.dataset.wider import wider

__sets = {}

def get_imdb(name, data_path=None):
    """Get an imdb (image database) by name."""
    try:
        __sets[name]
    except KeyError:
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name](data_path)

for split in ['train','val','test']:
    name = 'wider_{}'.format(split)
    __sets[name] = (lambda data_path, split=split: wider(split, data_path))