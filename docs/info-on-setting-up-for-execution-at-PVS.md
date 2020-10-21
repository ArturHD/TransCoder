# Setting up TransCoder for execution on Ubuntu

Artur Andrzejak, October 2020


## Setting up linux libs binaries and Python packages on Ubuntu

### Install Python package clang
+ For VM_ml4code the following works to update the "system" (non-venv) Python 3.6 interpreter: **pip install clang**
+ Note: use pip and not pip3. Updating the Python 3.6 SDK via intellijidea did not worked.


### Setting up llvm binaries for hyper/vm_ml4code

#### Add SW repos for linux (llvm)
See https://apt.llvm.org/ for current libclang version(s).
Execute **sudo apt-add-repository "deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-11 main"**

And maybe (not needed): **sudo apt-add-repository "deb-src http://apt.llvm.org/xenial/ llvm-toolchain-xenial-11 main"**

#### Install linux packages
+ Check first the latest versions at https://apt.llvm.org/, there section "Install (stable branch)". Here we have llvm-11 (20.10.2020).
+ To install just clang, lld and lldb (11 release):
sudo apt-get install clang-11 lldb-11 lld-11

#### Identify lib dir and file name
After successful installation of clang-11, check where is the lib dir and which files are there:
cd /usr/lib/llvm-11/lib$ 
ll libclang*
gives among others:
 libclang-11.so.1 -> ../../x86_64-linux-gnu/libclang-11.so.1*

So we have:
=> the lib dir is "llvm-11"
=> the correct libclang file is "libclang-11.so.1".
Use these in setting up TransCoder code.

#### Update TransCoder settings for libclang-11.so library:
In TransCoder code in 
preprocessing/src/code_tokenizer.py, lines 24-26
add specification of the library path and file name as follows
according to the right path and file specs in the subsec above this one):

`	# clang.cindex.Config.set_library_path('/usr/lib/llvm-7/lib/')
	clang.cindex.Config.set_library_path('/usr/lib/llvm-11/lib/')
	clang.cindex.Config.set_library_file('libclang-11.so.1')
`

### Setting up llvm binaries for Colab
+ Colab has already llvm file libclang-6.0.so here: /usr/lib/llvm-6.0/lib/libclang-6.0.so.1
+ So we just need to change the settings in TC fork (in preprocessing/src/code_tokenizer.py, lines 24-26)
`	# clang.cindex.Config.set_library_path('/usr/lib/llvm-7/lib/')
	clang.cindex.Config.set_library_path('/usr/lib/llvm-6.0/lib/')
	clang.cindex.Config.set_library_file('libclang-6.0.so.1')
`



## Execute pretrained models

### Download the trained models and set paths
On VM_ml4code I downloaded the both trained models to
/projects/TransCoder/TrainedModels
via:
`wget https://dl.fbaipublicfiles.com/transcoder/model_1.pth
 wget https://dl.fbaipublicfiles.com/transcoder/model_2.pth`

Then I created a symbolic link in the project dir:
`cd ~/IdeaProjects/TransCoder/data
ln -s /projects/TransCoder/TrainedModels/ TrainedModels` 

so that the dir appears as:  
`~/IdeaProjects/TransCoder/data/TrainedModels`

### Tring translating
According to https://github.com/ArturHD/TransCoder readme/Translate, the following should work on pre-trained models:

Original:  
`python translate.py --src_lang cpp --tgt_lang java --model_path trained_model.pth < input_code.cpp `

If the working dir is /home/artur/IdeaProjects/TransCoder, then the parameters for an intellijidea run are:  
`--src_lang cpp --tgt_lang java --model_path data/TrainedModels/model_1.pth`

In addition, in intellijidea run params we must set the input redirection to std input from some cpp file.
> I created the directory TransCoder/data/tests_artur
> Put there primes.cpp from [here](https://www.programiz.com/cpp-programming/examples/prime-number)
> In run config settings, use the option "Redirect input from:" (bottom, "Execution") to `/home/artur/IdeaProjects/TransCoder/data/tests_artur/primes.cpp`.


## Fixing errors related to GPU and lack of a GPU
### Problem running translate.py
We get an error:
`AssertionError:   
Found no NVIDIA driver on your system. Please check that you have an NVIDIA GPU and installed a driver from http://www.nvidia.com/Download/index.aspx`

### Solution attempt A - set env variable
According to
[link](https://stackoverflow.com/questions/53266350/how-to-tell-pytorch-to-not-use-the-gpu), we can set an env variable indicating that there are no GPUs, and PyTorch will respect this.
+ Shell:   `export CUDA_VISIBLE_DEVICES=""`
+ Python / Python console in intellijidea:    `os.environ["CUDA_VISIBLE_DEVICES"]=""`

=> Not working.

### Solution attempt B - install PyTorch without CUDA

+ Ininstall current Pytorch (cmd line): `pip uninstall torch; pip uninstall torchvision`
+ Install PyTorch with CPU only:  
`pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html`
  (according to [here](https://pytorch.org/)).

### Attempt B - follow up errors

We get an error: `File "/home/artur/.local/lib/python3.6/site-packages/torch/cuda/__init__.py", line 243, in __enter__
    self.prev_idx = torch._C._cuda_getDevice()
AttributeError: module 'torch._C' has no attribute '_cuda_getDevice' `

#### Fixing line 188
The root cause is in XLM/src.model/__init__ line 183, where the function torch.load is loaded with arguments:
`torch.load(enc_path, map_location=lambda storage, loc: storage.cuda(params.local_rank))`
which indicates that the model should be loaded to a GPU with device number params.local_rank.
According to documentation of torch.load, we can use sth like:

`# Load all tensors onto the CPU
torch.load('tensors.pt', map_location=torch.device('cpu'))`

=> Changed the code in TransCoder to:  
`enc_reload = torch.load(enc_path, map_location=torch.device('cpu'))`

#### Fixing line 214
Same problem in XLM/src.model/__init__ line 214, changed to:
`dec_reload = torch.load(dec_path, map_location=torch.device('cpu'))`

#### Fixing line 124
Same problem in XLM/src.model/__init__ line 124, changed to:
`reloaded = torch.load(params.reload_model, map_location=torch.device('cpu'))['model']`


### Follow-up errors after fixing torch.load()
#### Fixing line 258
Problem in XLM/src.model/__init__ line 258:
Code `return [encoder.cuda()], [dec.cuda() for dec in decoders]`
triggers `AssertionError: Torch not compiled with CUDA enabled`.

Replaced by:
        `return [encoder.cpu()], [dec.cpu() for dec in decoders]`

#### Fixing line 95 in translate.py
Here we have 
`self.encoder.cuda()
 self.decoder.cuda()` 
 which triggers same assertion `AssertionError: Torch not compiled with CUDA enabled`.
Changed to:
        `self.encoder.cpu()
        self.decoder.cpu()`

#### Pre-success - finally!
After these changes, output is:
`Loading codes from /home/artur/IdeaProjects/TransCoder/data/BPE_with_comments_codes ...
Read 50000 codes from the codes file.`

So we probably need to redirect a c++ file to std input.

#### Another error after input redirect
In translate.py, line 124:
`len1 = torch.LongTensor(1).fill_(len1).to(DEVICE)`
we get again: `AssertionError: Torch not compiled with CUDA enabled`.

TODO: solve this...