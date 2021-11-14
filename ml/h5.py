import h5py
import numpy as np

with h5py.File("animals.h5", 'w') as f:
    f.create_dataset('animals_include', data=np.array(["dogs".encode(), "cats".encode()]))
    dogs_group = f.create_group("dogs")
    f.create_dataset('cats', data = np.array(np.random.randn(5,64,64,3)))
    dogs_group.create_dataset('husky', data = np.random.randn(64,64,3))
    dogs_group.create_dataset('shiba', data = np.random.randn(64,64,3))
    print("ok")
    
    
with h5py.File("animals.h5", 'r') as f:
    for fkey in f.keys():
        print(f[fkey], fkey)
        
    print("============================")
    
    dogs_group = f["dogs"]
    for dkey in dogs_group.keys():
        print(dkey,dogs_group[dkey], dogs_group[dkey].name, dogs_group[dkey].value)
