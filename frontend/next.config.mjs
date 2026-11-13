// next.config.mjs
const nextConfig = {
  output: 'export',        // <-- static HTML/JS/CSS, no Node server needed
  images: { unoptimized: true },
}
export default nextConfig