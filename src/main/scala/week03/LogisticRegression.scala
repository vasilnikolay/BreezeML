package week03

import breeze.linalg._
import scala.math.{E, pow}

trait LogisticRegression {

  def sigmoid(G: DenseMatrix[Double]): DenseMatrix[Double] = {
    def innerSigmoid(input: Double): Double = {
      1.0 /(1.0 + pow(E, -1.0 * input))
    }
    G.map(innerSigmoid)
  }

  def computeCost(X: DenseMatrix[Double], Y: DenseMatrix[Double], THETA: DenseMatrix[Double]): Double = ???

  def predict(X: DenseMatrix[Double], THETA: DenseMatrix[Double]): Double = ???


  val test = DenseMatrix((1.0, 2.0), (2.0, 5.0))
  print(sigmoid(test))
}
