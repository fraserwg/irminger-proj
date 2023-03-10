export RUN_SECTION="b"
export RUN_TYPE="control"

mkdir ./$RUN_SECTION$RUN_TYPE
cd ./$RUN_SECTION$RUN_TYPE

# Link the input .data files
mkdir input
cd ./input
ln -s ../../a$RUN_TYPE/input/* ./

# Link the input files
cd ..
ln -s ../../../../src/3d-mitgcm-models/input/* ./
ln -s ../../../../src/3d-mitgcm-models/build200m/mitgcmuv ./
rm data
rm job.sh
cp ../a$RUN_TYPE/data ./
cp ../a$RUN_TYPE/job.sh ./