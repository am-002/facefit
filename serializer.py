import xml.etree.ElementTree as et
import numpy as np

def serialize_model(model, out_file):
    root = et.Element('model')
    et.SubElement(root, 'n_landmarks').text = str(model.n_landmarks)
    et.SubElement(root, 'mean_shape').text = arr_to_str(model.mean_shape.points)

    for fc in model.fern_cascades:
        _fern_cascade_to_xml(fc, root)

    tree = et.ElementTree(root)
    tree.write(out_file)

def _fern_to_xml(fern, root):
    node = et.SubElement(root, 'fern')
    et.SubElement(node, 'features').text = arr_to_str(fern.features)
    et.SubElement(node, 'thresholds').text = arr_to_str(fern.thresholds)
    bins = et.SubElement(node, 'bins')

    for bin, coeff in zip(fern.compressed_bins, fern.compressed_coeffs):
        bin_node = et.SubElement(bins, 'bin')
        et.SubElement(bin_node, 'indices').text = arr_to_str(bin)
        et.SubElement(bin_node, 'coeffs').text = arr_to_str(coeff)
        # for i in xrange(len(bin)):
        #     et.SubElement(bin_node, 'coeff').text = arr_to_str([bin[i], coeff[i]])
    return node

def _fern_cascade_to_xml(fern_cascade, root):
    node = et.SubElement(root, 'fern_cascade')
    _feature_extractor_to_xml(fern_cascade.feature_extractor, node)
    basis_node = et.SubElement(node, 'basis')
    for base_vector in fern_cascade.basis:
        et.SubElement(basis_node, 'base_vector').text = arr_to_str(base_vector)
    ferns_node = et.SubElement(node, 'ferns')
    for fern in fern_cascade.ferns:
        _fern_to_xml(fern, ferns_node)

def _feature_extractor_to_xml(feature_extractor, root):
    node = et.SubElement(root, 'feature_extractor')
    et.SubElement(node, 'lmarks').text = arr_to_str(feature_extractor.lmark)
    et.SubElement(node, 'pixel_coordinates').text = arr_to_str(feature_extractor.pixel_coords)

    # coords_node.text = ""
    # for lmark, coords in zip(feature_extractor.lmark, feature_extractor.pixel_coords):
    #     coords_node.text += str(lmark) + " " + arr_to_str(coords)


def arr_to_str(arr):
    s = ""
    for x in np.array(arr).flatten():
        s += str(x)
        s += " "
    return s