# mozaik
Automatic reproduction of images with simple primitives

## Usage
```mozaik.py -i <inputfile> [-o <outputfile> -c <shapeCounts> -p <primitive>]```

## A few examples

```mozaik.py -i examples/pearl.jpg -c 500 -p triangle```
<table>
  <tr>
    <td><img src='https://github.com/eleurent/mozaik/raw/master/examples/pearl.jpg' width='400' /></td>
    <td><img src='https://github.com/eleurent/mozaik/raw/master/out/pearl_500_32042.png' width='400' /></td>
  </tr>
</table>


```mozaik.py -i examples/resistance.jpg -c 100 -p circle```
<table>
  <tr>
    <td><img src='https://github.com/eleurent/mozaik/raw/master/examples/resistance.jpg' width='400' /></td>
    <td><img src='https://github.com/eleurent/mozaik/raw/master/out/resistance_300_101733.png' width='400' /></td>
  </tr>
</table>


```mozaik.py -i examples/nighthawks.png -c 500 -p rectangle```
<table>
  <tr>
    <td><img src='https://github.com/eleurent/mozaik/raw/master/examples/nighthawks.png' width='400' /></td>
    <td><img src='https://github.com/eleurent/mozaik/raw/master/out/nighthawks_500_19699.png' width='400' /></td>
  </tr>
</table>


```mozaik.py -i examples/pika.jpg -c 200 -p triangle```
<table>
  <tr>
    <td><img src='https://github.com/eleurent/mozaik/raw/master/examples/pika.jpg' width='400' /></td>
    <td><img src='https://github.com/eleurent/mozaik/raw/master/out/pika_200_115731.png' width='400' /></td>
  </tr>
</table>


```mozaik.py -i examples/sharbat.jpg -c 200 -p ellipse```
<table>
  <tr>
    <td><img src='https://github.com/eleurent/mozaik/raw/master/examples/sharbat.jpg' width='400' /></td>
    <td><img src='https://github.com/eleurent/mozaik/raw/master/out/sharbat_200_28389.png' width='400' /></td>
  </tr>
</table>
