# BW_Images_Colorization
A image colorization project built using keras/tensorflow

## Data Collection

- First, download chrome driver from [here](https://chromedriver.chromium.org/downloads) , according to your chrome browser version.

- Type the following commands
```
pip install tqdm
pip install numpy
pip install pillow
pip install selenium
git clone https://github.com/ultralytics/google-images-download
cd google-images-download
python bing_scraper.py --search "topic to search" --limit 100 --download --chromedriver "C:\path\to\chromedriver.exe"
```

## Data Preprocessing 

```
python datapreprocessing.py
```
## Model Making

```
python model.py
```
## Testing

```
python predict.py
```
## Architecture Used

![B07777_08_24](https://user-images.githubusercontent.com/43717493/103136118-9702dd80-46e3-11eb-90d9-54f87fbfa817.png)

## Results

<table>
  <thead>
    <tr>
      <th>Input</th>
      <th>Output</th>
    </tr>
   </thead>
   <tbody>
     <tr>
       <td><img align="left" alt="" src="https://user-images.githubusercontent.com/43717493/104124997-03373f80-537a-11eb-945a-7baffa180d36.jpg" width="256px" height="256px" /></td>
       <td><img align="left" alt="" src="https://user-images.githubusercontent.com/43717493/104124983-f87caa80-5379-11eb-86de-16ff33111492.png" /></td>
     </tr>
  </tbody>
</table>

