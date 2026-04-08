import { useMemo, useState } from 'react'

const SCENARIOS = {
  'Clean Data': {
    noise_std: 0,
    mismatch: false,
    description: 'No added noise and no dynamics mismatch. Best case for a classical solver.',
  },
  'Noisy Data': {
    noise_std: 0.5,
    mismatch: false,
    description: 'Adds measurement noise while keeping the same underlying dynamics.',
  },
  'Model Mismatch': {
    noise_std: 0,
    mismatch: true,
    description:
      'Data follows time-varying dynamics while the classical solver assumes fixed rho=28.',
  },
}

const DIAGNOSTIC_MEANING = {
  heteroscedasticity: 'Residual variance changes with prediction level.',
  autocorrelation: 'Residuals are correlated over time.',
  non_stationary: 'Residual behavior drifts over time.',
  state_dependence: 'Residual size depends on the current state.',
}

function formatValue(value, digits = 6) {
  if (value === null || value === undefined) return 'N/A'
  if (typeof value !== 'number') return String(value)
  return value.toFixed(digits)
}

function App() {
  const [scenario, setScenario] = useState('Clean Data')
  const [apiBaseUrl, setApiBaseUrl] = useState(import.meta.env.VITE_API_BASE_URL || '')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [results, setResults] = useState(null)

  const scenarioConfig = SCENARIOS[scenario]

  const decisionMeta = useMemo(() => {
    if (!results) return null
    if (results.decision === 'classical_ok') {
      return {
        tone: 'success',
        text: 'Classical solver sufficient — residuals do not show strong failure patterns.',
      }
    }
    const failed = Object.entries(results?.diagnostics?.test_results || {})
      .filter(([, flagged]) => flagged)
      .map(([name]) => name)
    return {
      tone: 'warning',
      text:
        failed.length > 0
          ? `Neural ODE selected — flagged diagnostics: ${failed.join(', ')}.`
          : 'Neural ODE selected — classical dynamics are not sufficient for this case.',
    }
  }, [results])

  async function runSimulation() {
    setLoading(true)
    setError('')
    setResults(null)

    try {
      const base = apiBaseUrl.trim().replace(/\/$/, '')
      const endpoint = base ? `${base}/run` : '/run'

      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          noise_std: scenarioConfig.noise_std,
          mismatch: scenarioConfig.mismatch,
          save_plots: false,
          show_plots: false,
        }),
      })

      if (!response.ok) {
        throw new Error(`Request failed (${response.status}). Check backend endpoint and CORS settings.`)
      }

      const data = await response.json()
      setResults(data)
    } catch (err) {
      setError(err.message || 'Failed to run simulation.')
    } finally {
      setLoading(false)
    }
  }

  const diagnosticsRows = Object.entries(results?.diagnostics?.test_results || {}).map(([name, flagged]) => ({
    name,
    result: flagged ? 'Fail' : 'Pass',
    meaning: DIAGNOSTIC_MEANING[name] || '',
  }))

  return (
    <div className="page">
      <header>
        <h1>Adaptive ODE Solver</h1>
        <p>Minimal frontend for scenario-based diagnostics and model switching.</p>
      </header>

      <section className="panel">
        <div className="field">
          <label>Scenario</label>
          <select value={scenario} onChange={(e) => setScenario(e.target.value)}>
            {Object.keys(SCENARIOS).map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
          <small>{scenarioConfig.description}</small>
        </div>

        <div className="field">
          <label>Backend URL (optional)</label>
          <input
            type="text"
            placeholder="http://localhost:8000"
            value={apiBaseUrl}
            onChange={(e) => setApiBaseUrl(e.target.value)}
          />
          <small>Frontend sends POST to /run on this base URL.</small>
        </div>

        <button onClick={runSimulation} disabled={loading}>
          {loading ? 'Running...' : 'Run Simulation'}
        </button>
      </section>

      {error && <div className="banner error">{error}</div>}

      {results && (
        <>
          {decisionMeta && <div className={`banner ${decisionMeta.tone}`}>{decisionMeta.text}</div>}

          <section className="panel">
            <h2>Metrics</h2>
            <table>
              <thead>
                <tr>
                  <th>Metric</th>
                  <th>Classical</th>
                  <th>Neural ODE</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>MSE</td>
                  <td>{formatValue(results?.metrics_classical?.mse)}</td>
                  <td>{formatValue(results?.metrics_neural?.mse)}</td>
                </tr>
                <tr>
                  <td>RMSE</td>
                  <td>{formatValue(results?.metrics_classical?.rmse)}</td>
                  <td>{formatValue(results?.metrics_neural?.rmse)}</td>
                </tr>
                <tr>
                  <td>MAE</td>
                  <td>{formatValue(results?.metrics_classical?.mae)}</td>
                  <td>{formatValue(results?.metrics_neural?.mae)}</td>
                </tr>
              </tbody>
            </table>
            {results?.metrics_neural == null && (
              <p className="note">Neural ODE metrics are N/A because classical solver was selected.</p>
            )}

            {results?.improvement_percent != null && (
              <div className="kpi">Improvement: {formatValue(results.improvement_percent, 2)}%</div>
            )}
          </section>

          <section className="panel">
            <h2>Diagnostics</h2>
            <table>
              <thead>
                <tr>
                  <th>Test</th>
                  <th>Result</th>
                  <th>Meaning</th>
                </tr>
              </thead>
              <tbody>
                {diagnosticsRows.map((row) => (
                  <tr key={row.name}>
                    <td>{row.name}</td>
                    <td>{row.result}</td>
                    <td>{row.meaning}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>

          <section className="panel">
            <h2>Plots</h2>
            <p className="note">
              If your backend returns image URLs/base64 in <code>plots</code>, they will render below.
            </p>
            <div className="plot-grid">
              {['trajectory', 'residuals', 'model_comparison'].map((key) => {
                const value = results?.plots?.[key]
                const src = typeof value === 'string' ? value : null
                if (!src) return null
                return (
                  <figure key={key}>
                    <figcaption>{key}</figcaption>
                    <img src={src} alt={key} />
                  </figure>
                )
              })}
            </div>
          </section>
        </>
      )}
    </div>
  )
}

export default App
