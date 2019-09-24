package week03

import breeze.linalg._
import helper.getMatrixFromFile
import scala.math.{E, log, pow}


trait LogisticRegression {
  def sigmoid(G: DenseMatrix[Double]): DenseMatrix[Double] = {
    def innerSigmoid(input: Double): Double = {
      1.0 /(1.0 + pow(E, -1.0 * input))
    }
      G.map(innerSigmoid)
  }

  // return cost and gradient as (cost, grad)
  def computeCost(X: DenseMatrix[Double],
                  Y: DenseMatrix[Double],
                  THETA: DenseMatrix[Double]): (Double, DenseMatrix[Double]) = {
    val setSize: Int = X.rows
    val sigmoidVec = sigmoid(X * THETA)
    val ERR: DenseMatrix[Double] = sigmoidVec - Y
    val dJ = (ERR.t * X) .* (1.0 / setSize)
    val cost: Double = sum(
      Y.t * sigmoidVec.map(log) +
        Y.map(x => - 1 * x + 1).t * sigmoidVec.map(x => log(- 1 * x + 1))
      ) / (setSize * (-1.0))
    (cost, dJ)
  }


  def predict(X: DenseMatrix[Double], THETA: DenseMatrix[Double]): Double = {
    val inp: DenseMatrix[Double] = DenseMatrix(sum(X.t * THETA))
    sigmoid(inp)(0, 0)
  }


}
