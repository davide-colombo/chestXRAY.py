
import matplotlib.pyplot as plt
import seaborn as sns

class PlotUtils:

    def plot_class_histogram(self, class1, class2, class3, **kwargs):
        sns.set_style('white')
        plt.figure(figsize=(10, 10), dpi=80)
        sns.distplot(class1, color='dodgerblue', label= 'bacteria', **kwargs)
        sns.distplot(class2, color='orange', label='normal', **kwargs)
        sns.distplot(class3, color='deeppink', label='virus', **kwargs)
        # plt.hist(class1, **kwargs, color='g', label='bacteria')
        # plt.hist(class2, **kwargs, color='y', label='normal')
        # plt.hist(class3, **kwargs, color='r', label='virus')
        plt.title('Density Histogram of the Inverse of the Variance of Graylevel values')
        plt.ylabel('Density')
        plt.xlabel('Pixel Inverse of the Variance')
        plt.xlim(0, 250)
        plt.legend()
