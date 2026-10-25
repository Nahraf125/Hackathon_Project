"""Lightweight replacement for the face_recognition package.
Uses dlib + face_recognition_models directly (same models, same 128-d encodings),
so the app does not need the `face_recognition` package or a dlib source build."""
import dlib
import numpy as np
import face_recognition_models

_detector = dlib.get_frontal_face_detector()
_pose = dlib.shape_predictor(face_recognition_models.pose_predictor_five_point_model_location())
_rec = dlib.face_recognition_model_v1(face_recognition_models.face_recognition_model_location())


def face_locations(img, number_of_times_to_upsample=1):
    """Returns [(top, right, bottom, left), ...]"""
    h, w = img.shape[:2]
    out = []
    for r in _detector(img, number_of_times_to_upsample):
        out.append((max(r.top(), 0), min(r.right(), w), min(r.bottom(), h), max(r.left(), 0)))
    return out


def face_encodings(img, known_face_locations=None, num_jitters=1):
    if known_face_locations is None:
        known_face_locations = face_locations(img)
    encs = []
    for (t, r, b, l) in known_face_locations:
        shape = _pose(img, dlib.rectangle(l, t, r, b))
        encs.append(np.array(_rec.compute_face_descriptor(img, shape, num_jitters)))
    return encs


def face_distance(faces, face_to_compare):
    if len(faces) == 0:
        return np.empty((0,))
    return np.linalg.norm(np.array(faces) - face_to_compare, axis=1)
