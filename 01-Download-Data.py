
# coding: utf-8

# # Download data
# Firstly we download data for condition monitoring from zenodo at https://doi.org/10.5281/zenodo.1323611

# In[13]:


data_url='https://zenodo.org/record/1323611/files/data.zip?download=1'



# In[14]:


import os, requests, zipfile, io

def download_and_extract(url, destination, force=False):
    response = requests.get(url)
    zipDocument = zipfile.ZipFile(io.BytesIO(response.content))
    # Attempt to see if we are going to overwrite anything
    if not force:
        abort = False
        for file in zipDocument.filelist:
            if os.path.isfile(os.path.join(destination, file.filename)):
                print(file.filename, 'already exists. If you want to overwrite the file call the method with force=True')
                abort = True
        if abort:
            print('Zip file was not extracted')
            return
        
    zipDocument.extractall(destination)


# In[15]:


download_and_extract(data_url, 'Dataset/ZEMA_Hydraulic/')

