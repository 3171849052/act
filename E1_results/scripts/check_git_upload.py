"""Check staged Git blobs before publishing; never uploads ignored files."""
from pathlib import PurePosixPath
import subprocess

MAX_BYTES = 1024 * 1024
FORBIDDEN = {'.ckpt','.pt','.pth','.safetensors','.onnx','.h5','.hdf5',
             '.pkl','.pickle','.npy','.npz','.mp4','.avi','.mov','.mkv','.webm',
             '.zip','.tar','.tgz','.7z','.whl','.conda','.pem','.key'}

def main():
    entries = subprocess.check_output(['git','ls-files','--stage','-z']).decode().split('\0')
    errors=[]; total=0; count=0; largest=(0,'')
    for entry in entries:
        if not entry: continue
        fields,name=entry.split('\t',1)
        mode,sha,stage=fields.split()
        if stage!='0': errors.append('Unmerged file: '+name);continue
        size=int(subprocess.check_output(['git','cat-file','-s',sha]))
        path=PurePosixPath(name)
        if size>=MAX_BYTES: errors.append('File must be smaller than 1 MiB: '+name)
        if path.suffix.lower() in FORBIDDEN:errors.append('Excluded artifact type: '+name)
        if mode=='120000':errors.append('Machine-local symlink: '+name)
        if path.parts[0] in {'data','data_local','results','.opencode'}:errors.append('Local-only directory: '+name)
        if path.name=='.env' or path.name.startswith('.env.') and path.name!='.env.example':
            errors.append('Environment secrets file: '+name)
        count+=1;total+=size
        if size>largest[0]:largest=(size,name)
    if errors:raise SystemExit('\n'.join(errors))
    print('PASS:',count,'staged files;',total,'bytes total; largest:',largest[1],largest[0],'bytes')

if __name__=='__main__':main()
