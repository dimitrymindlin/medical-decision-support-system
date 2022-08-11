import tqdm
import numpy as np
import tensorflow as tf
import pylib as py
import imlib as im



def calculate_tcv_os(dataset, translation_name, G_A2B, G_B2A, clf, oracle, save_dir, print_images=True):
    len_dataset = 0
    translated_images = []
    y_pred_translated = []
    y_pred_oracle = []
    for img_batch in tqdm.tqdm(dataset):
        if translation_name == "A2B":
            first_genrator = G_A2B
            cycle_generator = G_B2A

        else:
            first_genrator = G_B2A
            cycle_generator = G_A2B


        img_transformed = first_genrator.predict(img_batch)
        img_transformed /= np.max(np.abs(img_transformed), axis=0) # scale between -1 and 1
        # Cycle
        img_cycled = cycle_generator.predict(img_transformed)
        img_cycled /= np.max(np.abs(img_cycled), axis=0)

        for img_i, translated_i, cycled_i in zip(img_batch, img_transformed, img_cycled):
            translated_images.append(tf.squeeze(translated_i))
            y_pred_translated.append(
                int(np.argmax(clf(tf.expand_dims(tf.image.resize(translated_i, [512, 512]), axis=0)))))
            y_pred_oracle.append(
                int(np.argmax(oracle(tf.expand_dims(translated_i, axis=0)))))

            if print_images:
                img = np.concatenate([img_i, translated_i, cycled_i], axis=1)
                img_name = translation_name + "_" + str(len_dataset) + ".png"
                im.imwrite(img, py.join(save_dir, img_name))

            len_dataset += 1

    if translation_name == "A2B":
        tcv = sum(y_pred_translated) / len_dataset
        similar_predictions_count = sum(x == y == 1 for x, y in zip(y_pred_translated, y_pred_oracle))
        os = (1 / len_dataset) * similar_predictions_count
    else:
        tcv = (len_dataset - sum(y_pred_translated)) / len_dataset
        similar_predictions_count = sum(x == y == 0 for x, y in zip(y_pred_translated, y_pred_oracle))
        os = (1 / len(y_pred_translated)) * similar_predictions_count

    print(f"TCV:", float("{0:.3f}".format(np.mean(tcv))))
    print(f"OS :", float("{0:.3f}".format(np.mean(os))))
    return tcv, os, translated_images