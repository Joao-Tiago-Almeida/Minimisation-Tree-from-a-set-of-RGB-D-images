from scipy.io import savemat
import pickle as pk

file = open( "point_clouds.p", "rb" )
dict_pc = pk.load( file )
file.close()

new={}

savemat( "allpointcloud.mat", {'world':dict_pc[0], 'two':dict_pc[2] } )
