
We use the [[ObservableFunction.md | ObservableFunction]] class
to manage observable functions.
To create an observable function,
you can simply provide a function that accepts a state and returns
the observable targets, along with its output dimension.
For example, a full-state observable function can be defined as follows:

```python
tmp_func = lambda x: x
observable_func = pkdl.ObservableFunction(tmp_func, 2)
```
