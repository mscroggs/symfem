# Adding an element to Symfem

If you simply want to experiment with elements, you can create a custom element by following the [custom element demo](demo/custom_element.py).

If you want to add an element to the Symfem library, this guide will tell you how to do this. After making these changes,
you should push them to a fork of Symfem and [open a pull request](CONTRIBUTING.md).

## Step 1: Adding an element file
To add an element to Symfem, you should create a new `.py` file in the folder [symfem/elements](symfem/elements). You should name this file
as your element's name in lowercase (with underscores instead of any spaces or dashes).

You can look at the files currently in the elements folder to see what format your file should take: your file should include
these important features:

- The documentation string at the start of the file should include DOI links to references where the element is defined
- Your element class must define an `__init__` method that takes `reference` as an input (it may optionally take additional arguments)
- Your element class should define `names` and `references` lists that give all the names that can be used to initialise this element
  and the names of all the reference cells the element is defined on. No two elements may use the same name on the same cell.
- Your element class can define `min_order` and/or `max_order` variables if there is a minimum or maximum degree/order that is valid
  for this element
- You element class should define the type of continuity that your element has, this can be:
  - `L2` (no continuity)
  - `C0` (continuous)
  - `C1` (continuous with continuous derivatives
  - `C{n}` (continuous with `{n}` continuous derivatives) (you must replace `{n}` with a natural number)
  - `H(div)` (continuous across facets in the normal direction)
  - `H(curl)` (continuous across facets in the tangential direction(s))
  - `inner H(div)` (continuous normal inner products across facets)
  - `inner H(curl)` (continuous tangential inner products across facets)
  - `integral inner H(div)` (continuous integrals of normal inner products across facets)


## Step 2: Adding the new element to the tests
Once you've added your element file, you need to add you element to the Symfem tests. You can do this by editing the file
[test/utils.py](test/utils.py). For each cell that your element is implemented on, you should add an entry to the dictionary
for that cell. These entries should have the format:

```python
        "YOUR_ELEMENT": [({"ARG": VALUE}, [1, 2]), ({}, range(3, 4))],
```

You should replace `YOUR_ELEMENT` with one of the values in the `names` list in your element file. For your element, you provide
a list of tuples that will used to generate tests: the first item in the tuple is a list of keyword arguments to be passed when
creating your element (for many elements, this dictionary will be empty), and the second item is a list or range of degrees/orders
to run the test on.


## Step 3: Running the tests
Once you've completed step 2, you can test that your element functions correctly by running (with `YOUR_ELEMENT` replaced by the
element name you used in step 2):

```bash
python3 -m pytest test/test_elements.py --elements-to-test YOUR_ELEMENT
```

## Step 4: Adding the new element to create.py
You must add the name of the new element (including any alternative names) to the docstring of the function `create_element`
in the file [symfem/create.py](symfem/create.py).

## Step 5: Adding the new element to README.md
You must add the name of the element to the README. You can do this by running:

```bash
python3 update_readme.py
```

## Step 6: Testing the documentation
To confirm that you have completes steps 4 and 5 correctly, you can run:

```bash
python3 -m pytest test/test_docs.py
```
