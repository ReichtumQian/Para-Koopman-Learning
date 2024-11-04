
class Factory:
  
  def __init__(self):
    self._creators = {}

  def register(self, name, creator):
    self._creators[name] = creator
  
  def create(self, name, *args, **kwargs):
    creator = self._creators.get(name)
    if not creator:
      raise ValueError(name)
    return creator(*args, **kwargs)


