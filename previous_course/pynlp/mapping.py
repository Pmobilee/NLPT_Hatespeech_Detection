import pandas as pd


# MAP ZAMPIERI TO KUMAR'S GUIDELINES
def zamp_to_kumar():
    zamp_df_train = pd.read_csv('E:\VU\subjectivity_mining\eigenBert\data\zampieri_normal\TrainData.csv', sep='\t')
    # zamp_df_test = pd.read_csv('data/kumar_fb/testData-tw.csv', sep='\t')
    labels_train = zamp_df_train['Label']
    #labels_test = zamp_df_test['Label']

    def mapping(labels):
        new_labels = []
        for label in labels:
            # Not offensive
            if label == 'Offensive':
                label = '0'
            # Offensive Targeted Individual
            elif label == 'non-offensive':
                label = '1'
            # Offensive targeted group
            elif label == 'OAG':
                label = 'aggressive'
            # Offensive targeted other
            elif label == 'OFF_TIN_OTH':
                label = 'Offensive'
            # Offensive Untargeted
            elif label == 'OFF_UNT_':
                label = 'Offensive'
            new_labels.append(label)
        return new_labels

    zamp_df_train['Label'] = mapping(labels_train)
    #zamp_df_test['Label'] = mapping(labels_test)
    zamp_df_train.to_csv('zampieri_test.csv', sep='\t', index=False)
    #zamp_df_test.to_csv('data/kumar_fb/testDatam-tw.csv', sep='\t', index=False)


    # MAP KUMAR TO ZAMPIERI'S GUIDELINES

# def kumar_to_zamp():
#         kumar_df_train = pd.read_csv('data/Kumar/trainData.csv', sep='\t')

#         # Using the Twitter testdata
#         kumar_df_test = pd.read_csv('data/Kumar/testData-tw.csv', sep='\t')
#         labels_train = kumar_df_train['Label']
#         labels_test = kumar_df_test['Label']

#         def mapping(labels):
#             new_labels = []
#             for label in labels:

#                 if label == 'NAG':
#                     label = ''
                
#                 elif label == 'OAG':
#                     label = ''
                
#                 elif label == 'CAG':
#                     label = ''

#                 new_labels.append(label)
#             return new_labels

#         kumar_df_train['Label'] = mapping(labels_train)
#         kumar_df_test['Label'] = mapping(labels_test)
#         kumar_df_train.to_csv('data/kumar/zamp_mapped_train.csv', sep='\t', index=False)
#         kumar_df_test.to_csv('data/kumar/zamp_mapped_test.csv', sep='\t', index=False)
    

zamp_to_kumar()