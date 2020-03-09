def moveFilesInDir(rootDir,outDir):
    for root, dirs, files in os.walk(rootDir):  # replace the . with your starting directory
        for file in files:
            path_file = os.path.join(root,file)
            shutil.copy2(path_file,outDir) # change you destination dir
            print(path_file)

# Visualize matrix
import matplotlib.cm as cm
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.imshow(mel_band, interpolation='bilinear', cmap=cm.Greys_r)
plt.show()