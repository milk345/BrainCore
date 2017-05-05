#coding:utf-8
import tensorflow as tf
import numpy as np
import xdrlib, sys
import xlrd
import os

def __init__(self):
    self=self




def make_layer(inputs, in_size, out_size, activate=None):
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    basis = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    result = tf.matmul(inputs, weights) + basis
    if activate is None:
        return result
    else:
        return activate(result)


class BPNeuralNetwork:
    brainCan = "./brainCan/";
    def __init__(self):
        self.session = tf.Session()
        self.loss = None
        self.optimizer = None
        self.input_n = 0
        self.hidden_n = 0
        self.hidden_size = []
        self.output_n = 0
        self.input_layer = None
        self.hidden_layers = []
        self.label_layer = None
        self.output_layer = None

    def __del__(self):
        self.session.close()

    def setup(self, input_number, hidden_size_array, output_number):
        # set size args
        self.input_n = input_number
        self.hidden_n = len(hidden_size_array)  # count of hidden layers
        self.hidden_size = hidden_size_array  # count of cells in each hidden layer
        self.output_n = output_number
        # build input layer
        self.input_layer = tf.placeholder(tf.float32, [None, self.input_n])
        # build label layer
        self.label_layer = tf.placeholder(tf.float32, [None, self.output_n])
        # build hidden layers
        in_size = self.input_n
        out_size = self.hidden_size[0]
        inputs = self.input_layer
        self.hidden_layers.append(make_layer(inputs, in_size, out_size, activate=tf.nn.relu))
        for i in range(self.hidden_n-1):
            in_size = out_size
            out_size = self.hidden_size[i+1]
            inputs = self.hidden_layers[-1]
            self.hidden_layers.append(make_layer(inputs, in_size, out_size, activate=tf.nn.relu))
        # build output layer
        self.output_layer = make_layer(self.hidden_layers[-1], self.hidden_size[-1], self.output_n)
        initer = tf.global_variables_initializer()
        # do training
        self.session.run(initer)

    def train(self, cases, labels, limit=10000, learn_rate=0.05):
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.square((self.label_layer - self.output_layer)), reduction_indices=[1]))
        self.optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(self.loss)
        # initer = tf.global_variables_initializer()
        # # do training
        # self.session.run(initer)
        for i in range(limit):
            self.session.run(self.optimizer, feed_dict={self.input_layer: cases, self.label_layer: labels})
            print(self.session.run(self.loss, feed_dict={self.input_layer: cases, self.label_layer: labels}))

    def predict(self, case):
        return self.session.run(self.output_layer, feed_dict={self.input_layer: case})

    def trainWrapper(self):
        x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y_data = np.array([[0, 1, 1, 0]]).transpose()        #  即[[0],[1],[1],[0]]
        test_data = np.array([[0, 1]])
        self.setup(2, [10, 5], 1)
        self.train(x_data, y_data)
        print (self.predict(test_data))



    def save(self):
        saver = tf.train.Saver()
        saver_path = saver.save(self.session, "C:/Users/Administrator/PycharmProjects/tensor/models/model.ckpt")
        print(saver_path)

    def restore(self):
        with tf.Session() as sess:
         saver = tf.train.Saver()
         saver.restore(sess, "C:/Users/Administrator/PycharmProjects/tensor/models/model.ckpt")

    def save_with_id(self,id):

        if not os.path.exists(self.brainCan+str(id)+"milk"):
          os.system("sudo mkdir -p "+self.brainCan+str(id)+"milk");
          # os.makedirs(self.brainCan+str(id)+"milk")
        saver = tf.train.Saver()
        saver_path = saver.save(self.session, self.brainCan+str(id)+"milk/brain.ckpt")
        print(saver_path)

    def restore_with_id(self,id):
        with tf.Session() as sess:
         saver = tf.train.Saver()
         saver.restore(self.session, self.brainCan+str(id)+"milk/brain.ckpt")

    # def test(self):
    #     test_data = np.array([[0, 1]])
    #     print(self.predict(test_data))

class God:


         def createByGod(self,shape,input_number,output_number,brain_id):


            brain = BPNeuralNetwork()
            brain.setup(input_number,shape,output_number)
            brain.save_with_id(brain_id)
            return "ok"


         def praticeByGod(self,pratice_data_address,label_data_address,shape,input_number,output_number,brain_id):


            brain=BPNeuralNetwork()
            brain.setup(input_number, shape, output_number)
            brain.restore_with_id(brain_id)

            x_data =self.get_table_from_excel(pratice_data_address)
            y_data =self.get_table_from_excel(label_data_address)

            if not(x_data[0].size==input_number)or not(y_data[0].size==output_number):
                return"wrong_format"
            brain.train(x_data,y_data)
            return "ok"

         def predictByGod(self, input_array,shape, input_number,output_number,brain_id):
            inputs=np.array(input_array)

            if not(inputs.size==input_number):
                return"wrong_format"
            brain=BPNeuralNetwork()
            brain.setup(input_number, shape, output_number)
            brain.restore_with_id(brain_id)
            return brain.predict(inputs)[0]

          # 获取训练数据
         @staticmethod
         def open_excel(file):
            try:
                data = xlrd.open_workbook(file)
                return data
            except  Exception as e:
                print(str(e))


         # 根据索引获取Excel表格中的数据   参数:file：Excel文件路径     colnameindex：表头列名所在行的所以  ，by_index：表的索引
         def get_table_from_excel(self, file_address, colnameindex=0, by_index=0):
            data = self.open_excel(file_address)
            table = data.sheets()[by_index]
            nrows = table.nrows  # 行数
            ncols = table.ncols  # 列数
            colnames = table.row_values(colnameindex)  # 获得某一行数据，这里是第一行
            # multiply_row = np.array(ndmin=2)
            multiply_row = np.arange(0)
            print(multiply_row.shape)
            for rownum in range(1, nrows):

                row = table.row_values(rownum)
                if row:
                    one_row = np.arange(0)
                    for i in range(len(colnames)):
                        temp = np.array([row[i]])
                        # print(temp)
                        one_row = np.concatenate((one_row, temp), axis=0)
                    one_row = one_row[np.newaxis, :]
                    # print("又一行"+str(one_row))
                    if multiply_row.shape != (0,):
                        multiply_row = np.concatenate((multiply_row, one_row), axis=0)  # 连接第一个维度，即外面的中括号
                    else:
                        multiply_row = one_row

            return multiply_row













#
#
# def createByJava(self,shape,input_output_id_list):
#
#     input_number = input_output_id_list[0]
#     output_number = input_output_id_list[1]
#     brain_id = input_output_id_list[2]
#
#     brain = BPNeuralNetwork()
#     brain.setup(input_number,turn_list_into_array(shape),output_number)
#     brain.save_with_id(id)
#     return "ok"
#
#
# def praticeByJava(self,pratice_data_address,label_data_address,shape,input_output_id_list):
#
#
#     input_number=input_output_id_list[0]
#     output_number=input_output_id_list[1]
#     brain_id=input_output_id_list[2]
#
#     brain=BPNeuralNetwork()
#     brain.setup(input_number, turn_list_into_array(shape), output_number)
#     brain.restore_with_id(brain_id)
#
#     x_data = get_table_from_excel(pratice_data_address)
#     y_data = get_table_from_excel(label_data_address)
#     brain.train(x_data,y_data)
#     return "ok"
#
#
#
# def predictByJava(self, input_list,shape, input_output_id_list):
#
#     input_number = input_output_id_list[0]
#     output_number = input_output_id_list[1]
#     brain_id = input_output_id_list[2]
#
#     brain=BPNeuralNetwork()
#     brain.setup(input_number, turn_list_into_array(shape), output_number)
#     brain.restore_with_name(id)
#     return brain.predict(turn_list_into_array(input_list))
#
#
# def turn_list_into_array(self,list):
#
#     array=np.array()
#     for i in range(len(list)):
#         item=list[i]    #在这里认为list是个数组，可是传过来的是pyList对象哦
#         if item:
#           new_member=np.array([item])
#           array=np.concatenate((array,new_member),axis=0)
#
#     return array







# def askForTest():
#     option = input("will you test this brain?:\n")
#     if option=="y":
#         nn.test()
#         print("ok!")
#         askForTest()
#     else:
#         askForTest()
#
# def askForTrain():
#     option = input("will you train this brain?:\n")
#     if option=="y":
#         nn.trainWrapper()
#         print("ok!")
#         askForStore()
#     else:
#         askForRestore()
#
# def askForStore():
#     option = input("will you store your model?:\n")
#     if option=="y":
#         nn.save()
#         print("已保存，请放心退出")
#     else:
#         askForRestore()
#
# def askForRestore():
#     option = input("will you restore your model?:\n")
#     if option=="y":
#         nn.restore()
#         print("ok!")
#         askForTest()
#     else:
#         askForStore()




