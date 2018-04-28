package core

import "math"

//LinearModel 线性模型
type LinearModel struct {
	Model Matrix
}

//LinearRegression 线性回归
type LinearRegression struct{}

/*
 * 使用线性回归模型对线性数据进行建模
 *  trainData:训练数据
 *  trainLabel:训练数据对应的标签
 */
func (linear LinearRegression) Fit(trainData, trainLabel Matrix) LinearModel {
	theta := linear.fit(trainData, trainLabel, 0.01, 10000)
	rows, cols := theta.Shape()
	var model Matrix = make([][]float64, rows)
	for i := 0; i < rows; i++ {
		model[i] = make([]float64, cols)
		{
			for j := 0; j < cols; j++ {
				model[i][j] = theta[i][j]
			}
		}
	}
	return LinearModel{Model: model}
}

// Predicate 使用模型进行预测
func (model LinearModel) Predicate(predicateData Matrix) Matrix {
	return predicateData.Dot(model.Model)
}

//Score 模型得分
func (model LinearModel) Score(testLabel, predicateResult Matrix) float64 {
	rows,_:= testLabel.Shape()
	return math.Sqrt(predicateResult.MinusMatrix(testLabel).Power().Sum()) / float64(rows)
}

//fit 使用线性模型对现有数据进行线性回归
func (linear LinearRegression) fit(trainData, trainLabel Matrix, alpha float64, iteratorSteps int) Matrix {
	rows, cols := trainData.Shape()
	// 在所有数据前面加一列
	data := Ones(rows, cols+1)
	MatrixCopy(data, 1, trainData, 0)
	//构建theta
	theta := Zeros(cols+1, 1)
	return gradientDescending(data, trainLabel, theta, alpha, iteratorSteps)
}

//gradientDescending 梯度下降算法
func gradientDescending(trainData, trainLabel, theta Matrix, alpha float64, iteratorSteps int) Matrix {
	m, _ := trainData.Shape()
	xt := trainData.Transform()
	for i := 0; i < iteratorSteps; i++ {
		h := trainData.Dot(theta)
		g := xt.Dot(h.MinusMatrix(trainLabel)).Multiply(alpha)
		mi, err := g.DivideValue(float64(m))
		if err != nil {
			panic(err.Error())
		}
		theta = theta.MinusMatrix(mi)
	}
	return theta
}
