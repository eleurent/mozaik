# mozaik
Automatic reproduction of images with simple primitives

## Usage
```mozaik.py -i <inputfile> [-o <outputfile> -c <shapeCounts> -p <primitive> -m <method> -s <seed>]```

## A few examples

```mozaik.py -i examples/pearl.jpg -c 500 -p triangle```
<table>
  <tr>
    <td><img src='examples/pearl.jpg' width='400' /></td>
    <td><img src='out/pearl_500_32042.png' width='400' /></td>
  </tr>
</table>


```mozaik.py -i examples/resistance.jpg -c 100 -p circle```
<table>
  <tr>
    <td><img src='examples/resistance.jpg' width='400' /></td>
    <td><img src='out/resistance_300_101733.png' width='400' /></td>
  </tr>
</table>


```mozaik.py -i examples/nighthawks.png -c 500 -p rectangle```
<table>
  <tr>
    <td><img src='examples/nighthawks.png' width='400' /></td>
    <td><img src='out/nighthawks_500_19699.png' width='400' /></td>
  </tr>
</table>


```mozaik.py -i examples/pika.jpg -c 200 -p triangle```
<table>
  <tr>
    <td><img src='examples/pika.jpg' width='400' /></td>
    <td><img src='out/pika_200_115731.png' width='400' /></td>
  </tr>
</table>


```mozaik.py -i examples/sharbat.jpg -c 200 -p ellipse```
<table>
  <tr>
    <td><img src='examples/sharbat.jpg' width='400' /></td>
    <td><img src='out/sharbat_200_28389.png' width='400' /></td>
  </tr>
</table>

## Credits

This project is more than largely inspired by *Michael Fogleman*'s excellent [primitive](https://github.com/fogleman/primitive).
