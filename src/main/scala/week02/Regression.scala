package week02


import breeze.linalg.{DenseMatrix, DenseVector, sum}
import helper.getMatrixFromFile


trait Regression {


  def computeCostFunction(X: DenseMatrix[Double],
                          Y: DenseMatrix[Double],
                          THETA: DenseMatrix[Double]): Double = {
    val n = Y.rows
    sum((X * THETA - Y).t * (X * THETA - Y))/(2*n)
  }


  def gradientDescent(X: DenseMatrix[Double],
                      Y: DenseMatrix[Double],
                      THETA: DenseMatrix[Double],
                      alpha: Double,
                      numIter: Int): DenseMatrix[Double] = {

    val J_history = DenseVector.zeros[Double](numIter)
    J_history(0) = computeCostFunction(X, Y, THETA)
    val setSize = X.cols
    var INNER_THETA = THETA
    var theta0c = 0.0
    var theta1c = 0.0
    var theta0next = 0.0
    var theta1next = 0.0
    var J = 0.0
    for (i <- 1 until numIter){
      theta0c = INNER_THETA(0, 0)
      theta1c = INNER_THETA(1, 0)
      INNER_THETA = INNER_THETA - (alpha/setSize) * ((X * INNER_THETA - Y).t * X).t
      // theta0next = theta0c - (alpha/setSize) * sum(X * INNER_THETA - Y)
      // theta1next = theta1c - (alpha/setSize) * sum((X * INNER_THETA - Y).t * X(0 to -1, 1 to 1))
      J_history(i) = computeCostFunction(X, Y, INNER_THETA)
    }
    INNER_THETA
  }


  val MAT: DenseMatrix[Double] = getMatrixFromFile("w02ex1data1.txt")
  val iterations = 1500
  val alpha = 0.0001
  val THETA: DenseMatrix[Double] = DenseMatrix.zeros[Double](2,1)
  val X: DenseMatrix[Double] = DenseMatrix.horzcat(DenseMatrix.ones[Double](MAT.rows, 1), MAT(0 to -1, 0 to 0))
  val Y: DenseMatrix[Double] = MAT(0 to -1, 1 to 1)

  // in case of THETA = (0,0) expression returns 32.07
  println("Jo = %f".format(computeCostFunction(X, Y, THETA)))
  // run gradient descent
  val t: DenseMatrix[Double] = gradientDescent(X, Y, THETA, alpha, iterations)
  println("The result of gradient descent: h(x) = %f + %f * x".format(t(0,0), t(1,0)))
}

