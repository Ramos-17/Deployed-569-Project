import { useEffect, useMemo, useState } from 'react'
import axios from 'axios'
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceDot,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import './App.css'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000',
})

const multiplierMarks = [
  { value: 0.5, label: '0.5x' },
  { value: 1, label: '1.0x' },
  { value: 1.5, label: '1.5x' },
  { value: 2, label: '2.0x' },
]

function formatCountryLabel(country) {
  return country.country_name
    ? `${country.country_name} (${country.country_id})`
    : country.country_id
}

function App() {
  const [countries, setCountries] = useState([])
  const [selectedCountryId, setSelectedCountryId] = useState('')
  const [selectedBaseYear, setSelectedBaseYear] = useState('')
  const [co2Multiplier, setCo2Multiplier] = useState(1)
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(true)
  const [predicting, setPredicting] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    let active = true

    async function loadOptions() {
      try {
        setLoading(true)
        setError('')
        const response = await api.get('/options')
        if (!active) return

        const options = response.data.countries ?? []
        setCountries(options)

        if (options.length > 0) {
          const firstCountry = options[0]
          setSelectedCountryId(firstCountry.country_id)
          setSelectedBaseYear(String(firstCountry.base_years.at(-1)))
        }
      } catch (requestError) {
        if (!active) return
        setError(
          requestError.response?.data?.detail ??
            'Unable to load available countries. Make sure the FastAPI server is running and the dataset is in place.',
        )
      } finally {
        if (active) {
          setLoading(false)
        }
      }
    }

    loadOptions()
    return () => {
      active = false
    }
  }, [])

  const selectedCountry = useMemo(
    () => countries.find((country) => country.country_id === selectedCountryId) ?? null,
    [countries, selectedCountryId],
  )

  const availableBaseYears = useMemo(
    () => selectedCountry?.base_years ?? [],
    [selectedCountry],
  )

  const effectiveBaseYear = useMemo(() => {
    if (!selectedCountry || availableBaseYears.length === 0) return ''
    const current = Number(selectedBaseYear)
    return availableBaseYears.includes(current)
      ? String(current)
      : String(availableBaseYears.at(-1))
  }, [availableBaseYears, selectedBaseYear, selectedCountry])

  const chartData = useMemo(() => {
    if (!prediction) return []

    const history = prediction.historical.map((item) => ({
      year: item.year,
      historical: item.anomaly,
      predicted: null,
    }))

    return [
      ...history,
      {
        year: prediction.target_year,
        historical: null,
        predicted: prediction.predicted_anomaly,
      },
    ]
  }, [prediction])

  async function handleSubmit(event) {
    event.preventDefault()
    if (!selectedCountryId || !effectiveBaseYear) return

    try {
      setPredicting(true)
      setError('')
      const targetYear = Number(effectiveBaseYear) + 1
      const response = await api.post('/predict', {
        country_id: selectedCountryId,
        target_year: targetYear,
        co2_multiplier: Number(co2Multiplier),
      })
      setPrediction(response.data)
    } catch (requestError) {
      setPrediction(null)
      setError(
        requestError.response?.data?.detail ??
          'Prediction failed. Verify the model, scaler, dataset, and feature list are aligned.',
      )
    } finally {
      setPredicting(false)
    }
  }

  return (
    <main className="app-shell">
      <section className="hero-panel">
        <p className="eyebrow">Climate LSTM Forecast Studio</p>
        <h1>Stress-test next year&apos;s temperature anomaly with a CO2 shock slider.</h1>
        <p className="hero-copy">
          Select a country, anchor the forecast on five historical years, and compare the
          model&apos;s next-year anomaly prediction against the observed trend.
        </p>
      </section>

      <section className="workspace">
        <form className="control-panel" onSubmit={handleSubmit}>
          <div className="panel-heading">
            <h2>Scenario Controls</h2>
            <p>Base year determines the last observed year. The app predicts the following year.</p>
          </div>

          <label className="field">
            <span>Country</span>
            <select
              value={selectedCountryId}
              onChange={(event) => setSelectedCountryId(event.target.value)}
              disabled={loading || countries.length === 0}
            >
              {countries.map((country) => (
                <option key={country.country_id} value={country.country_id}>
                  {formatCountryLabel(country)}
                </option>
              ))}
            </select>
          </label>

          <label className="field">
            <span>Base Year</span>
            <select
              value={effectiveBaseYear}
              onChange={(event) => setSelectedBaseYear(event.target.value)}
              disabled={loading || availableBaseYears.length === 0}
            >
              {availableBaseYears.map((year) => (
                <option key={year} value={year}>
                  {year}
                </option>
              ))}
            </select>
          </label>

          <label className="field slider-field">
            <div className="slider-copy">
              <span>CO2 Emission Impact</span>
              <strong>{Number(co2Multiplier).toFixed(2)}x</strong>
            </div>
            <input
              type="range"
              min="0.5"
              max="2"
              step="0.05"
              value={co2Multiplier}
              onChange={(event) => setCo2Multiplier(event.target.value)}
              disabled={loading}
            />
            <div className="slider-scale">
              {multiplierMarks.map((mark) => (
                <span key={mark.value}>{mark.label}</span>
              ))}
            </div>
          </label>

          <button type="submit" className="predict-button" disabled={loading || predicting}>
            {predicting ? 'Running Forecast...' : 'Predict Next Year'}
          </button>

          {error ? <p className="status error">{error}</p> : null}
          {!error && loading ? <p className="status">Loading country coverage...</p> : null}
          {!error && !loading && prediction ? (
            <p className="status">
              Forecast ready for {prediction.country_name ?? prediction.country_id}: predicted anomaly
              for {prediction.target_year} is{' '}
              <strong>{prediction.predicted_anomaly.toFixed(3)}</strong>.
            </p>
          ) : null}
        </form>

        <section className="chart-panel">
          <div className="panel-heading">
            <h2>Historical Window + Forecast</h2>
            <p>
              Historical points show the five observed anomalies. The final gold marker is the
              model&apos;s predicted anomaly for the next year.
            </p>
          </div>

          <div className="chart-frame">
            {prediction ? (
              <ResponsiveContainer width="100%" height={420}>
                <LineChart data={chartData} margin={{ top: 20, right: 28, bottom: 12, left: 4 }}>
                  <CartesianGrid stroke="rgba(130, 167, 177, 0.18)" strokeDasharray="4 4" />
                  <XAxis dataKey="year" tickLine={false} axisLine={false} />
                  <YAxis tickLine={false} axisLine={false} width={56} />
                  <Tooltip
                    formatter={(value) =>
                      value === null || value === undefined ? '--' : Number(value).toFixed(3)
                    }
                    labelFormatter={(label) => `Year ${label}`}
                  />
                  <Line
                    type="monotone"
                    dataKey="historical"
                    stroke="#0b7285"
                    strokeWidth={3}
                    dot={{ r: 4, fill: '#0b7285' }}
                    connectNulls={false}
                    name="Historical anomaly"
                  />
                  <Line
                    type="monotone"
                    dataKey="predicted"
                    stroke="#f59f00"
                    strokeWidth={3}
                    dot={{ r: 6, fill: '#f59f00', stroke: '#fff7e6', strokeWidth: 3 }}
                    connectNulls={false}
                    name="Predicted anomaly"
                  />
                  <ReferenceDot
                    x={prediction.target_year}
                    y={prediction.predicted_anomaly}
                    r={10}
                    fill="#f59f00"
                    stroke="#fff7e6"
                    strokeWidth={3}
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="empty-state">
                <p>Run a forecast to plot the five-year history and the next-year prediction.</p>
              </div>
            )}
          </div>
        </section>
      </section>
    </main>
  )
}

export default App
