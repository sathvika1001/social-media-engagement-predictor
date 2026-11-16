import { useEffect, useState } from 'react'

const POST_TYPES = ['image', 'video', 'carousel', 'poll', 'text']
const DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
const PLATFORMS = ['Facebook', 'Instagram', 'Twitter']
const HOURS = Array.from({ length: 24 }, (_, i) => i)

function Bars({ title, items }) {
  return (
    <div style={{ marginBottom: 24 }}>
      <h3>{title}</h3>
      {items.map((item) => (
        <div key={item.label} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
          <span style={{ width: 70 }}>{item.label}</span>
          <div style={{ flex: 1, background: '#eee', borderRadius: 4 }}>
            <div style={{ width: `${item.pct}%`, background: '#4f46e5', height: 14, borderRadius: 4 }} />
          </div>
          <span style={{ width: 60, textAlign: 'right' }}>{item.value}</span>
        </div>
      ))}
    </div>
  )
}

export default function App() {
  const [dashboard, setDashboard] = useState(null)
  const [form, setForm] = useState({ post_type: 'image', post_day: 'Mon', post_hour: 12, platform: 'Facebook' })
  const [predicted, setPredicted] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    fetch('/api/dashboard')
      .then((res) => res.json())
      .then(setDashboard)
      .catch(() => setError('Could not load dashboard. Is Flask running on port 5000?'))
  }, [])

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value })
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setPredicted(null)
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...form, post_hour: Number(form.post_hour) }),
    })
    const data = await res.json()
    if (res.ok) {
      setPredicted(data.predicted_engagement)
    } else {
      setError(data.error)
    }
  }

  return (
    <div style={{ maxWidth: 800, margin: '0 auto', padding: 24, fontFamily: 'Inter, sans-serif' }}>
      <h1>Engagement Lab</h1>

      <section>
        <h2>Predict engagement</h2>
        <form onSubmit={handleSubmit} style={{ display: 'grid', gap: 12, maxWidth: 300 }}>
          <label>
            Platform
            <select name="platform" value={form.platform} onChange={handleChange}>
              {PLATFORMS.map((p) => <option key={p}>{p}</option>)}
            </select>
          </label>
          <label>
            Post type
            <select name="post_type" value={form.post_type} onChange={handleChange}>
              {POST_TYPES.map((t) => <option key={t}>{t}</option>)}
            </select>
          </label>
          <label>
            Day
            <select name="post_day" value={form.post_day} onChange={handleChange}>
              {DAYS.map((d) => <option key={d}>{d}</option>)}
            </select>
          </label>
          <label>
            Hour
            <select name="post_hour" value={form.post_hour} onChange={handleChange}>
              {HOURS.map((h) => <option key={h} value={h}>{h}:00</option>)}
            </select>
          </label>
          <button type="submit">Predict</button>
        </form>

        {predicted !== null && (
          <p><strong>Predicted engagement: {predicted}</strong></p>
        )}
        {error && <p style={{ color: 'crimson' }}>{error}</p>}
      </section>

      <section style={{ marginTop: 32 }}>
        <h2>Your post data</h2>
        {!dashboard ? (
          <p>Loading...</p>
        ) : (
          <>
            <p>Total posts: {dashboard.total_posts}</p>
            <p>Average engagement: {dashboard.avg_engagement}</p>
            <p>Best post type: {dashboard.best_post_type}</p>
            <p>Best hour: {dashboard.best_hour}:00</p>
            <Bars title="By post type" items={dashboard.by_type} />
            <Bars title="By day" items={dashboard.by_day} />
            <Bars title="By hour" items={dashboard.by_hour} />
          </>
        )}
      </section>
    </div>
  )
}